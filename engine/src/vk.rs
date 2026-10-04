//! Streaming Vulkan compute backend for matvec.
//!
//! Architecture:
//!   - One Vulkan device, one compute queue, one command pool
//!   - Fused dequant+dot shader per quant type
//!   - Per-operation staging: allocate buffers, dispatch, readback, destroy
//!   - No persistent GPU state beyond device + pipeline objects

use crate::model::StreamedTensor;
use ash::vk;

// Include auto-generated SPIR-V data and pipeline loader
include!(concat!(env!("OUT_DIR"), "/spv/pipelines.rs"));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuQuantFormat {
    F32 = 0,
    F16 = 1,
    Q8_0 = 2,
    Q4K = 3,
}

impl GpuQuantFormat {
    pub fn from_ggml(dtype: u32) -> Option<Self> {
        use crate::gguf::*;
        match dtype {
            GGML_F32 => Some(Self::F32),
            GGML_F16 => Some(Self::F16),
            GGML_Q8_0 => Some(Self::Q8_0),
            GGML_Q4_K => Some(Self::Q4K),
            _ => None,
        }
    }
}

pub struct VulkanBackend {
    _instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    compute_queue: vk::Queue,
    command_pool: vk::CommandPool,
    pipelines: [Option<vk::Pipeline>; 4],
    pipeline_layout: vk::PipelineLayout,
    desc_pool: vk::DescriptorPool,
    desc_set_layout: vk::DescriptorSetLayout,
}

#[repr(C)]
struct MatvecParams {
    row_elems: u32,
    row_bytes: u32,
    n_rows: u32,
    start_row: u32,
}

fn spv_to_words(spv: &[u8]) -> Vec<u32> {
    assert!(spv.len().is_multiple_of(4));
    spv.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

unsafe fn create_host_buffer(
    device: &ash::Device,
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
    size: vk::DeviceSize,
) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer = unsafe {
        device
            .create_buffer(
                &vk::BufferCreateInfo {
                    size,
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )
            .unwrap()
    };
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };
    let mut mem_type = !0u32;
    for i in 0..mem_props.memory_type_count {
        if (reqs.memory_type_bits & (1 << i)) != 0
            && mem_props.memory_types[i as usize].property_flags.contains(
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        {
            mem_type = i;
            break;
        }
    }
    let memory = unsafe {
        device
            .allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: reqs.size,
                    memory_type_index: mem_type,
                    ..Default::default()
                },
                None,
            )
            .unwrap()
    };
    unsafe {
        device.bind_buffer_memory(buffer, memory, 0).unwrap();
    }
    (buffer, memory)
}

impl VulkanBackend {
    /// Create a new Vulkan backend, selecting the first discrete GPU.
    pub fn new() -> Result<Self, String> {
        let entry = unsafe { ash::Entry::load().map_err(|e| format!("ash::Entry::load: {e}"))? };
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&std::ffi::CString::new("slow-inference").unwrap())
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&std::ffi::CString::new("slow-inference").unwrap())
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0))
            .build();
        let layer_names: [*const i8; 0] = [];
        let ext_names: [*const i8; 0] = [];
        let instance = unsafe {
            entry
                .create_instance(
                    &vk::InstanceCreateInfo {
                        p_application_info: &app_info,
                        enabled_layer_count: layer_names.len() as u32,
                        pp_enabled_layer_names: layer_names.as_ptr(),
                        enabled_extension_count: ext_names.len() as u32,
                        pp_enabled_extension_names: ext_names.as_ptr(),
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| format!("create_instance: {e}"))?
        };
        let phys_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| format!("enumerate: {e}"))?;
        let (physical_device, qf) = phys_devices
            .iter()
            .filter_map(|&pd| {
                let _props = unsafe { instance.get_physical_device_properties(pd) };
                let qprops = unsafe { instance.get_physical_device_queue_family_properties(pd) };
                for (i, qp) in qprops.iter().enumerate() {
                    if qp.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                        return Some((pd, i as u32));
                    }
                }
                None
            })
            .next()
            .ok_or("no compute queue found")?;

        let queue_priorities = [1.0f32];
        let device = unsafe {
            instance
                .create_device(
                    physical_device,
                    &vk::DeviceCreateInfo {
                        queue_create_info_count: 1,
                        p_queue_create_infos: &vk::DeviceQueueCreateInfo {
                            queue_family_index: qf,
                            queue_count: 1,
                            p_queue_priorities: queue_priorities.as_ptr(),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| format!("create_device: {e}"))?
        };
        let compute_queue = unsafe { device.get_device_queue(qf, 0) };

        let command_pool = unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo {
                        queue_family_index: qf,
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| format!("command_pool: {e}"))?
        };

        // Descriptor set layout: SSBO bindings 0 (weights), 1 (act), 2 (out)
        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];
        let desc_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo {
                        binding_count: 3,
                        p_bindings: bindings.as_ptr(),
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| format!("desc_set_layout: {e}"))?
        };

        // Pipeline layout with push constants
        let pc_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<MatvecParams>() as u32,
        };
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo {
                        set_layout_count: 1,
                        p_set_layouts: &desc_set_layout,
                        push_constant_range_count: 1,
                        p_push_constant_ranges: &pc_range,
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| format!("pipeline_layout: {e}"))?
        };

        // Descriptor pool: 1 set
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 3,
        }];
        let desc_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo {
                        pool_size_count: 1,
                        p_pool_sizes: pool_sizes.as_ptr(),
                        max_sets: 1,
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| format!("desc_pool: {e}"))?
        };

        Ok(VulkanBackend {
            _instance: instance,
            device,
            physical_device,
            compute_queue,
            command_pool,
            pipelines: [None, None, None, None],
            pipeline_layout,
            desc_pool,
            desc_set_layout,
        })
    }

    pub fn load_pipeline(&mut self, fmt: GpuQuantFormat, spv: &[u8]) -> Result<(), String> {
        let idx = fmt as usize;
        if self.pipelines[idx].is_some() {
            return Ok(());
        }
        let words = spv_to_words(spv);
        let shader_module = unsafe {
            self.device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        code_size: words.len() * 4,
                        p_code: words.as_ptr(),
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| format!("ShaderModule: {e}"))?
        };
        let stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: c"main".as_ptr(),
            ..Default::default()
        };
        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo {
                        stage,
                        layout: self.pipeline_layout,
                        ..Default::default()
                    }],
                    None,
                )
                .map_err(|e| format!("CreatePipeline: {e:?}"))
                .unwrap()[0]
        };
        unsafe { self.device.destroy_shader_module(shader_module, None) };
        self.pipelines[idx] = Some(pipeline);
        Ok(())
    }

    /// Execute a GPU matvec: y = W · x
    ///
    /// W is streamed from disk as raw Q4/Q8 bytes, uploaded per-chunk to GPU.
    /// Returns None if this format has no pipeline loaded.
    /// Real block size per 256-element block, matching quant.rs dequant.
    /// GGUF type_layout may report something else (144 for Q4K in newer spec).
    /// We use the dequant's actual block size always.
    fn gpu_block_bytes(dtype: u32) -> Option<usize> {
        use crate::gguf::*;
        match dtype {
            GGML_F32 => Some(4),
            GGML_Q8_0 => Some(34),  // 2B scale + 32B quads
            GGML_Q4_K => Some(144), // 2B d + 2B dmin + 12B scales + 128B quants
            _ => None,
        }
    }

    /// Real block elements per block, matching quant.rs.
    fn gpu_block_elems(dtype: u32) -> Option<usize> {
        use crate::gguf::*;
        match dtype {
            GGML_F32 => Some(1),
            GGML_Q8_0 => Some(32),
            GGML_Q4_K => Some(256),
            _ => None,
        }
    }

    pub fn matvec_gpu(
        &mut self,
        dtype: u32,
        row_elems: usize,
        n_rows: usize,
        w: &StreamedTensor,
        x: &[f32],
    ) -> Option<Vec<f32>> {
        let fmt = GpuQuantFormat::from_ggml(dtype)?;
        let pipeline = self.pipelines[fmt as usize]?;
        if n_rows == 0 {
            return Some(vec![]);
        }
        let block_bytes = Self::gpu_block_bytes(dtype)?;
        let block_elems = Self::gpu_block_elems(dtype)?;
        // Compute actual row_bytes from the dequant block size, NOT from type_layout.
        // Type_layout may be outdated (144 vs 84 for Q4K).
        let nblk = row_elems / block_elems;
        let gpu_row_bytes = nblk * block_bytes;
        // Let's call it strap-on-row-bytes for the GPU
        eprintln!(
            "[vk] matvec dtype={dtype} fmt={fmt:?} n_rows={n_rows} row_elems={row_elems} file_row_bytes={} gpu_row_bytes={gpu_row_bytes} block_bytes={block_bytes}",
            w.row_bytes
        );
        let mut y = vec![0.0f32; n_rows];
        let mut scratch = Vec::new();
        let xs = (x.len() * 4) as vk::DeviceSize;
        let ys = (y.len() * 4) as vk::DeviceSize;

        unsafe {
            let a_buf_mem =
                create_host_buffer(&self.device, &self._instance, self.physical_device, xs);
            let o_buf_mem =
                create_host_buffer(&self.device, &self._instance, self.physical_device, ys);
            let (a_buf, a_mem) = a_buf_mem;
            let (o_buf, o_mem) = o_buf_mem;

            // Upload activation once
            let aptr = self
                .device
                .map_memory(a_mem, 0, xs, vk::MemoryMapFlags::empty())
                .ok()?;
            std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, aptr as *mut u8, x.len() * 4);
            self.device.unmap_memory(a_mem);

            // Allocate one descriptor set (reused for all chunks)
            let desc_set = self
                .device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                    descriptor_pool: self.desc_pool,
                    descriptor_set_count: 1,
                    p_set_layouts: &self.desc_set_layout,
                    ..Default::default()
                })
                .ok()?[0];

            // Chunk through rows — upload weights per chunk
            let max_weight_bytes = (64 * 1024) as vk::DeviceSize;
            let chunk_rows = (max_weight_bytes as usize / gpu_row_bytes).max(1);
            let mut start = 0usize;
            while start < n_rows {
                let nr = chunk_rows.min(n_rows - start);
                // Read raw bytes per-row using file row_bytes stride,
                // then take gpu_row_bytes from each (handles Q4K 84 vs 144 mismatch).
                let mut bytes = Vec::new();
                for r in 0..nr {
                    if let Ok(row_data) = w.raw_rows(start + r, 1, &mut scratch) {
                        bytes.extend_from_slice(&row_data[..gpu_row_bytes.min(row_data.len())]);
                    } else {
                        eprintln!("[vk] raw_rows failed at row {}", start + r);
                        return None;
                    }
                }

                let ws = bytes.len() as vk::DeviceSize;
                let (w_buf, w_mem) =
                    create_host_buffer(&self.device, &self._instance, self.physical_device, ws);

                // Upload weights
                let wptr = self
                    .device
                    .map_memory(w_mem, 0, ws, vk::MemoryMapFlags::empty())
                    .ok()?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), wptr as *mut u8, bytes.len());
                self.device.unmap_memory(w_mem);

                // Update weight buffer descriptor
                let wbinfo = vk::DescriptorBufferInfo {
                    buffer: w_buf,
                    offset: 0,
                    range: ws,
                };
                let abinfo = vk::DescriptorBufferInfo {
                    buffer: a_buf,
                    offset: 0,
                    range: xs,
                };
                let obinfo = vk::DescriptorBufferInfo {
                    buffer: o_buf,
                    offset: 0,
                    range: ys,
                };
                let writes = [
                    vk::WriteDescriptorSet {
                        dst_set: desc_set,
                        dst_binding: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: &wbinfo,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: desc_set,
                        dst_binding: 1,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: &abinfo,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: desc_set,
                        dst_binding: 2,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: &obinfo,
                        ..Default::default()
                    },
                ];
                self.device.update_descriptor_sets(&writes, &[]);

                let params = MatvecParams {
                    row_elems: row_elems as u32,
                    row_bytes: gpu_row_bytes as u32,
                    n_rows: nr as u32,
                    start_row: start as u32,
                };

                let cmd_buf = self
                    .device
                    .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                        command_pool: self.command_pool,
                        level: vk::CommandBufferLevel::PRIMARY,
                        command_buffer_count: 1,
                        ..Default::default()
                    })
                    .ok()?[0];

                self.device
                    .begin_command_buffer(cmd_buf, &vk::CommandBufferBeginInfo::default())
                    .ok()?;
                self.device
                    .cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[desc_set],
                    &[],
                );
                self.device.cmd_push_constants(
                    cmd_buf,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    std::slice::from_raw_parts(
                        &params as *const MatvecParams as *const u8,
                        std::mem::size_of::<MatvecParams>(),
                    ),
                );
                self.device
                    .cmd_dispatch(cmd_buf, nr.div_ceil(64) as u32, 1, 1);
                self.device.end_command_buffer(cmd_buf).ok()?;

                let fence = self
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .ok()?;
                self.device
                    .queue_submit(
                        self.compute_queue,
                        &[vk::SubmitInfo::builder()
                            .command_buffers(&[cmd_buf])
                            .build()],
                        fence,
                    )
                    .ok()?;
                self.device.wait_for_fences(&[fence], true, u64::MAX).ok()?;
                self.device.destroy_fence(fence, None);

                // Readback
                let optr = self
                    .device
                    .map_memory(o_mem, 0, ys, vk::MemoryMapFlags::empty())
                    .ok()?;
                std::ptr::copy_nonoverlapping(optr as *const u8, y.as_mut_ptr() as *mut u8, nr * 4);
                self.device.unmap_memory(o_mem);

                self.device
                    .free_command_buffers(self.command_pool, &[cmd_buf]);
                self.device.free_memory(w_mem, None);
                self.device.destroy_buffer(w_buf, None);

                start += nr;
            }

            self.device.free_memory(a_mem, None);
            self.device.destroy_buffer(a_buf, None);
            self.device.free_memory(o_mem, None);
            self.device.destroy_buffer(o_buf, None);
        }

        Some(y)
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            for &p in self.pipelines.iter().flatten() {
                self.device.destroy_pipeline(p, None);
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.desc_set_layout, None);
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}
