#define VK_NO_PROTOTYPES
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <vk_mem_alloc.h>
#include <volk.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

import Allocator;
import Buffer;
import Command;
import CommandPool;
import DescriptorBuffer;
import Device;
import Instance;
import Pipeline;
import Shader;
import Sync;

class DescriptorBindingTest : public testing::TestWithParam<bool> {};

TEST_P(DescriptorBindingTest, RepeatedAndSkippedBindsUseRequestedFrame) {
  RenderGraph::Instance instance("DescriptorBindingTest", true);
  RenderGraph::Device device(instance);
  RenderGraph::DeviceRequirements requirements;
  if (GetParam()) requirements.optionalExtensions = {VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME};
  device.initialize(requirements);
  if (GetParam() && !device.isExtensionEnabled(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) {
    GTEST_SKIP() << "VK_EXT_descriptor_buffer is unavailable";
  }

  RenderGraph::MemoryAllocator allocator(device, instance);
  std::array<std::unique_ptr<RenderGraph::Buffer>, 4> buffers;
  for (auto& buffer : buffers) {
    buffer = std::make_unique<RenderGraph::Buffer>(
        sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, allocator, device);
    const std::uint32_t zero = 0;
    ASSERT_EQ(vmaCopyMemoryToAllocation(allocator.getAllocator(), &zero, buffer->getAllocation(), 0, sizeof(zero)),
              VK_SUCCESS);
  }

  std::ifstream file(std::filesystem::path(renderGraphTestShaderDir) / "descriptorBinding.comp.spv",
                     std::ios::ate | std::ios::binary);
  ASSERT_TRUE(file.is_open());
  std::vector<char> code(static_cast<std::size_t>(file.tellg()));
  file.seekg(0);
  ASSERT_TRUE(file.read(code.data(), static_cast<std::streamsize>(code.size())));
  RenderGraph::Shader shader(device);
  shader.add(code);
  RenderGraph::DescriptorSetLayout layout(device);
  layout.createCustom(shader.getDescriptorSetLayoutBindings().front());
  std::vector<RenderGraph::DescriptorSetLayout*> layouts{&layout};
  RenderGraph::DescriptorPool pool({}, device);
  RenderGraph::Pipeline pipeline(device);
  pipeline.createCompute(shader.getShaderStageInfo().front(), layouts, {});

  std::unique_ptr<RenderGraph::DescriptorHandler> materialA;
  std::unique_ptr<RenderGraph::DescriptorHandler> materialB;
  if (GetParam()) {
    materialA = std::make_unique<RenderGraph::DescriptorBuffer>(layouts, allocator, device);
    materialB = std::make_unique<RenderGraph::DescriptorBuffer>(layouts, allocator, device);
  } else {
    materialA = std::make_unique<RenderGraph::DescriptorSet>(layouts, pool, device);
    materialB = std::make_unique<RenderGraph::DescriptorSet>(layouts, pool, device);
  }
  // A has three frame slots; B shares one slot across all frames.
  for (std::size_t frame = 0; frame < 3; ++frame) materialA->add({buffers[frame].get()});
  materialB->add({buffers[3].get()});

  RenderGraph::CommandPool commandPool(RenderGraph::QueueType::GENERAL, device);
  RenderGraph::CommandBuffer commands(commandPool, device);
  commands.beginCommands();
  materialA->initialize(commands);
  materialB->initialize(commands);
  EXPECT_THROW(materialA->bind(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.getPipelineLayout(), commands, 3),
               std::out_of_range);
  EXPECT_THROW(materialB->bind(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.getPipelineLayout(), commands, 1),
               std::out_of_range);

  vkCmdBindPipeline(commands.getCommandBuffer(), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.getPipeline());
  const auto dispatch = [&](RenderGraph::DescriptorHandler& material, std::uint32_t frame) {
    material.bind(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.getPipelineLayout(), commands, frame);
    vkCmdDispatch(commands.getCommandBuffer(), 1, 1, 1);
    const VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    };
    const VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(commands.getCommandBuffer(), &dependency);
  };

  // Skip A, then select slots out of order and rebind A within each frame.
  dispatch(*materialB, 0);
  for (std::uint32_t frame : {2u, 0u, 2u, 1u, 0u}) {
    dispatch(*materialA, frame);
    dispatch(*materialB, 0);
    dispatch(*materialA, frame);
  }
  dispatch(*materialB, 0);

  RenderGraph::Barrier readback;
  for (auto& buffer : buffers) {
    readback.addBuffer(buffer.get(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
  }
  readback.execute(commands.getCommandBuffer());
  commands.endCommands();
  const VkCommandBufferSubmitInfo commandInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = commands.getCommandBuffer(),
  };
  const VkSubmitInfo2 submit{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &commandInfo,
  };
  const auto queue = device.getQueue(RenderGraph::QueueType::GENERAL);
  ASSERT_EQ(vkQueueSubmit2(queue, 1, &submit, nullptr), VK_SUCCESS);
  ASSERT_EQ(vkQueueWaitIdle(queue), VK_SUCCESS);

  const std::array<std::uint32_t, 4> expected{4, 2, 4, 7};
  for (std::size_t index = 0; index < buffers.size(); ++index) {
    std::uint32_t value = 0;
    ASSERT_EQ(
        vmaCopyAllocationToMemory(allocator.getAllocator(), buffers[index]->getAllocation(), 0, &value, sizeof(value)),
        VK_SUCCESS);
    EXPECT_EQ(value, expected[index]) << "buffer " << index;
  }
}

INSTANTIATE_TEST_SUITE_P(Backends, DescriptorBindingTest, testing::Values(false, true));
