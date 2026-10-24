module;

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vk_mem_alloc.h>
#include <volk.h>

module ResourceUploader;

import Command;

using namespace RenderGraph;

ResourceUploader::UploadBatch::UploadBatch(const CommandPool& commandPool, const Device& device)
    : _commandBuffer(std::make_unique<CommandBuffer>(commandPool, device)) {}

ResourceUploader::ResourceUploader(const MemoryAllocator& memoryAllocator, const Device& device, QueueType queueType)
    : _memoryAllocator(&memoryAllocator),
      _device(&device),
      _commandPool(queueType, device),
      _timelineSemaphore(VK_SEMAPHORE_TYPE_TIMELINE, device) {}

ResourceUploader::UploadBatch& ResourceUploader::_getRecordingBatch() {
  if (_recordingBatch == nullptr) {
    _recordingBatch = std::make_unique<UploadBatch>(_commandPool, *_device);
    _recordingBatch->_commandBuffer->beginCommands();
  }
  return *_recordingBatch;
}

void ResourceUploader::_recordBufferBarrier(VkCommandBuffer commandBuffer,
                                            VkBuffer buffer,
                                            VkDeviceSize offset,
                                            VkDeviceSize size,
                                            VkPipelineStageFlags2 sourceStageMask,
                                            VkAccessFlags2 sourceAccessMask,
                                            VkPipelineStageFlags2 destinationStageMask,
                                            VkAccessFlags2 destinationAccessMask) {
  const VkBufferMemoryBarrier2 barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = sourceStageMask,
      .srcAccessMask = sourceAccessMask,
      .dstStageMask = destinationStageMask,
      .dstAccessMask = destinationAccessMask,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = buffer,
      .offset = offset,
      .size = size,
  };
  const VkDependencyInfo dependencyInfo{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &barrier,
  };
  vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

void ResourceUploader::upload(Buffer& destination, std::span<const std::byte> data, VkDeviceSize offset) {
  const VkDeviceSize dataSize = static_cast<VkDeviceSize>(data.size());
  if (offset > destination.getSize() || dataSize > destination.getSize() - offset) {
    throw std::out_of_range("Buffer upload range exceeds destination size");
  }
  if (data.empty()) {
    return;
  }

  VkMemoryPropertyFlags memoryPropertyFlags = 0;
  vmaGetAllocationMemoryProperties(_memoryAllocator->getAllocator(), destination.getAllocation(), &memoryPropertyFlags);

  const bool persistentlyMapped = destination.getAllocationInfo().pMappedData != nullptr;
  if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 && persistentlyMapped) {
    const VkResult result = vmaCopyMemoryToAllocation(_memoryAllocator->getAllocator(), data.data(),
                                                      destination.getAllocation(), offset, data.size());
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Can't upload data to host-visible buffer: " + std::to_string(result));
    }

    // Create a batch and begin its command buffer for the subsequent submit().
    _getRecordingBatch();
    return;
  }

  auto stagingBuffer = std::make_unique<Buffer>(
      dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, *_memoryAllocator,
      *_device);
  const VkResult copyResult = vmaCopyMemoryToAllocation(_memoryAllocator->getAllocator(), data.data(),
                                                        stagingBuffer->getAllocation(), 0, data.size());
  if (copyResult != VK_SUCCESS) {
    throw std::runtime_error("Can't upload data to staging buffer: " + std::to_string(copyResult));
  }

  UploadBatch& batch = _getRecordingBatch();
  const VkBuffer stagingBufferHandle = stagingBuffer->getBuffer();
  batch._stagingBuffers.push_back(std::move(stagingBuffer));
  const VkCommandBuffer commandBuffer = batch._commandBuffer->getCommandBuffer();
  _recordBufferBarrier(commandBuffer, stagingBufferHandle, 0, dataSize, VK_PIPELINE_STAGE_2_HOST_BIT,
                       VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
  _recordBufferBarrier(commandBuffer, destination.getBuffer(), offset, dataSize, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT);

  const VkBufferCopy copyRegion{
      .srcOffset = 0,
      .dstOffset = offset,
      .size = dataSize,
  };
  vkCmdCopyBuffer(commandBuffer, stagingBufferHandle, destination.getBuffer(), 1, &copyRegion);
}

void ResourceUploader::upload(Image& destination,
                              std::span<const std::byte> data,
                              std::span<const VkDeviceSize> bufferOffsets) {
  if (data.empty()) {
    return;
  }
  const bool invalidOffset = std::ranges::any_of(
      bufferOffsets, [dataSize = data.size()](VkDeviceSize offset) { return offset >= dataSize; });
  if (bufferOffsets.size() != static_cast<std::size_t>(destination.getLayerNumber()) || invalidOffset ||
      (destination.getUsageFlags() & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0) {
    throw std::invalid_argument("Invalid image upload");
  }
  const auto queueFlags = _device->getQueueFamilyProperties(_commandPool.getType()).queueFlags;
  if ((queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0 && destination.getAspectMask() != VK_IMAGE_ASPECT_COLOR_BIT) {
    throw std::invalid_argument("Non-color image uploads require a GENERAL queue");
  }
  if ((queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0 &&
      std::ranges::any_of(bufferOffsets, [](VkDeviceSize offset) { return offset % 4 != 0; })) {
    throw std::invalid_argument("Transfer-only image uploads require four-byte aligned buffer offsets");
  }
  std::vector<VkBufferImageCopy> copyRegions;
  copyRegions.reserve(bufferOffsets.size());
  // Whole mip-level copies satisfy the queue family's minImageTransferGranularity, including (0, 0, 0).
  for (std::size_t layerIndex = 0; layerIndex < bufferOffsets.size(); ++layerIndex) {
    copyRegions.push_back({
        .bufferOffset = bufferOffsets[layerIndex],
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = destination.getAspectMask(),
                .mipLevel = 0,
                .baseArrayLayer = static_cast<uint32_t>(layerIndex),
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent =
            {
                static_cast<uint32_t>(destination.getResolution().x),
                static_cast<uint32_t>(destination.getResolution().y),
                1,
            },
    });
  }

  auto stagingBuffer = std::make_unique<Buffer>(
      static_cast<VkDeviceSize>(data.size()), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, *_memoryAllocator,
      *_device);
  const VkResult copyResult = vmaCopyMemoryToAllocation(_memoryAllocator->getAllocator(), data.data(),
                                                        stagingBuffer->getAllocation(), 0, data.size());
  if (copyResult != VK_SUCCESS) {
    throw std::runtime_error("Can't upload image data to staging buffer: " + std::to_string(copyResult));
  }

  UploadBatch& batch = _getRecordingBatch();
  const VkBuffer stagingBufferHandle = stagingBuffer->getBuffer();
  batch._stagingBuffers.push_back(std::move(stagingBuffer));
  const VkCommandBuffer commandBuffer = batch._commandBuffer->getCommandBuffer();
  _recordBufferBarrier(commandBuffer, stagingBufferHandle, 0, static_cast<VkDeviceSize>(data.size()),
                       VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT);

  if (destination.getImageLayout() != VK_IMAGE_LAYOUT_GENERAL) {
    Barrier transition;
    transition.addImage(destination.getImage(), 0, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        destination.getImageLayout(), VK_IMAGE_LAYOUT_GENERAL,
                        {destination.getAspectMask(), 0, static_cast<uint32_t>(destination.getMipMapNumber()), 0,
                         static_cast<uint32_t>(destination.getLayerNumber())});
    transition.execute(commandBuffer);
    destination.setImageLayout(VK_IMAGE_LAYOUT_GENERAL);
  } else {
    const VkImageMemoryBarrier2 beforeCopyBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = destination.getImage(),
        .subresourceRange =
            {
                .aspectMask = destination.getAspectMask(),
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = static_cast<uint32_t>(destination.getLayerNumber()),
            },
    };
    const VkDependencyInfo dependencyInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &beforeCopyBarrier,
    };
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
  }

  vkCmdCopyBufferToImage(commandBuffer, stagingBufferHandle, destination.getImage(), VK_IMAGE_LAYOUT_GENERAL,
                         static_cast<uint32_t>(copyRegions.size()), copyRegions.data());
}

GpuTicket ResourceUploader::submit() {
  if (_recordingBatch == nullptr) {
    throw std::logic_error("Can't submit an empty upload batch");
  }

  const VkQueue queue = _device->getQueue(_commandPool.getType());
  _inFlightBatches.push_back(std::move(_recordingBatch));
  UploadBatch& batch = *_inFlightBatches.back();
  batch._commandBuffer->endCommands();
  const VkCommandBufferSubmitInfo commandBufferInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = batch._commandBuffer->getCommandBuffer(),
  };
  const VkSemaphoreSubmitInfo signalSemaphoreInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = _timelineSemaphore.getSemaphore(),
      .value = _nextCompletionValue,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
  };
  const VkSubmitInfo2 submitInfo{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &commandBufferInfo,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &signalSemaphoreInfo,
  };

  const VkResult result = vkQueueSubmit2(queue, 1, &submitInfo, nullptr);
  if (result != VK_SUCCESS) {
    _inFlightBatches.pop_back();
    throw std::runtime_error("Resource upload submission failed: " + std::to_string(result));
  }
  batch._completionValue = _nextCompletionValue++;
  return {
      .semaphore = _timelineSemaphore.getSemaphore(),
      .value = batch._completionValue,
  };
}

void ResourceUploader::reclaim() {
  if (_inFlightBatches.empty()) {
    return;
  }

  uint64_t completedValue = 0;
  const VkResult result = vkGetSemaphoreCounterValue(_device->getLogicalDevice(), _timelineSemaphore.getSemaphore(),
                                                     &completedValue);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Can't query resource upload completion: " + std::to_string(result));
  }

  while (!_inFlightBatches.empty() && _inFlightBatches.front()->_completionValue <= completedValue) {
    _inFlightBatches.pop_front();
  }
}

ResourceUploader::~ResourceUploader() {
  if (!_inFlightBatches.empty()) {
    const uint64_t completionValue = _inFlightBatches.back()->_completionValue;
    const VkSemaphore semaphore = _timelineSemaphore.getSemaphore();
    const VkSemaphoreWaitInfo waitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &semaphore,
        .pValues = &completionValue,
    };
    vkWaitSemaphores(_device->getLogicalDevice(), &waitInfo, std::numeric_limits<uint64_t>::max());
  }
}
