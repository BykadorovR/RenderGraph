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

ResourceUploader::ResourceUploader(const MemoryAllocator& memoryAllocator, const Device& device)
    : _memoryAllocator(&memoryAllocator),
      _device(&device),
      _commandPool(QueueType::GRAPHICS, device),
      _timelineSemaphore(VK_SEMAPHORE_TYPE_TIMELINE, device) {}

ResourceUploader::UploadBatch& ResourceUploader::_getRecordingBatch() {
  if (_recordingBatch == nullptr) {
    auto commandBuffer = std::make_unique<CommandBuffer>(_commandPool, *_device);
    commandBuffer->beginCommands();

    _recordingBatch = std::make_unique<UploadBatch>();
    _recordingBatch->_commandBuffer = std::move(commandBuffer);
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

  if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
    const VkResult result = vmaCopyMemoryToAllocation(_memoryAllocator->getAllocator(), data.data(),
                                                      destination.getAllocation(), offset, data.size());
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Can't upload data to host-visible buffer: " + std::to_string(result));
    }

    const VkCommandBuffer commandBuffer = _getRecordingBatch()._commandBuffer->getCommandBuffer();
    _recordBufferBarrier(commandBuffer, destination.getBuffer(), offset, dataSize, VK_PIPELINE_STAGE_2_HOST_BIT,
                         VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    return;
  }

  auto stagingBuffer = std::make_unique<Buffer>(
      dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, *_memoryAllocator);
  const VkResult copyResult = vmaCopyMemoryToAllocation(_memoryAllocator->getAllocator(), data.data(),
                                                        stagingBuffer->getAllocation(), 0, data.size());
  if (copyResult != VK_SUCCESS) {
    throw std::runtime_error("Can't upload data to staging buffer: " + std::to_string(copyResult));
  }

  UploadBatch& batch = _getRecordingBatch();
  const VkCommandBuffer commandBuffer = batch._commandBuffer->getCommandBuffer();
  _recordBufferBarrier(commandBuffer, stagingBuffer->getBuffer(), 0, dataSize, VK_PIPELINE_STAGE_2_HOST_BIT,
                       VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
  _recordBufferBarrier(commandBuffer, destination.getBuffer(), offset, dataSize, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT);

  const VkBufferCopy copyRegion{
      .srcOffset = 0,
      .dstOffset = offset,
      .size = dataSize,
  };
  vkCmdCopyBuffer(commandBuffer, stagingBuffer->getBuffer(), destination.getBuffer(), 1, &copyRegion);

  _recordBufferBarrier(commandBuffer, destination.getBuffer(), offset, dataSize, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
  batch._stagingBuffers.push_back(std::move(stagingBuffer));
}

void ResourceUploader::_generateMipmaps(Image& image, VkCommandBuffer commandBuffer) {
  VkImageMemoryBarrier2 barrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image.getImage(),
      .subresourceRange =
          {
              .aspectMask = image.getAspectMask(),
              .levelCount = 1,
              .baseArrayLayer = 0,
              .layerCount = static_cast<uint32_t>(image.getLayerNumber()),
          },
  };
  const VkDependencyInfo dependencyInfo{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier,
  };

  int mipWidth = image.getResolution().x;
  int mipHeight = image.getResolution().y;
  for (uint32_t mipLevel = 1; mipLevel < static_cast<uint32_t>(image.getMipMapNumber()); ++mipLevel) {
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.subresourceRange.baseMipLevel = mipLevel - 1;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

    const VkImageBlit blit{
        .srcSubresource =
            {
                .aspectMask = image.getAspectMask(),
                .mipLevel = mipLevel - 1,
                .baseArrayLayer = 0,
                .layerCount = static_cast<uint32_t>(image.getLayerNumber()),
            },
        .srcOffsets = {{0, 0, 0}, {mipWidth, mipHeight, 1}},
        .dstSubresource =
            {
                .aspectMask = image.getAspectMask(),
                .mipLevel = mipLevel,
                .baseArrayLayer = 0,
                .layerCount = static_cast<uint32_t>(image.getLayerNumber()),
            },
        .dstOffsets = {{0, 0, 0}, {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1}},
    };
    vkCmdBlitImage(commandBuffer, image.getImage(), VK_IMAGE_LAYOUT_GENERAL, image.getImage(), VK_IMAGE_LAYOUT_GENERAL,
                   1, &blit, VK_FILTER_LINEAR);

    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

    mipWidth = mipWidth > 1 ? mipWidth / 2 : 1;
    mipHeight = mipHeight > 1 ? mipHeight / 2 : 1;
  }

  barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.subresourceRange.baseMipLevel = static_cast<uint32_t>(image.getMipMapNumber() - 1);
  vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
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
  if (destination.getMipMapNumber() > 1 &&
      (destination.getAspectMask() != VK_IMAGE_ASPECT_COLOR_BIT ||
       (destination.getUsageFlags() & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0)) {
    throw std::invalid_argument("Image does not support mipmap generation");
  }

  auto stagingBuffer = std::make_unique<Buffer>(
      static_cast<VkDeviceSize>(data.size()), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, *_memoryAllocator);
  const VkResult copyResult = vmaCopyMemoryToAllocation(_memoryAllocator->getAllocator(), data.data(),
                                                        stagingBuffer->getAllocation(), 0, data.size());
  if (copyResult != VK_SUCCESS) {
    throw std::runtime_error("Can't upload image data to staging buffer: " + std::to_string(copyResult));
  }

  UploadBatch& batch = _getRecordingBatch();
  const VkCommandBuffer commandBuffer = batch._commandBuffer->getCommandBuffer();
  _recordBufferBarrier(commandBuffer, stagingBuffer->getBuffer(), 0, static_cast<VkDeviceSize>(data.size()),
                       VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT);

  if (destination.getImageLayout() != VK_IMAGE_LAYOUT_GENERAL) {
    destination.changeLayout(destination.getImageLayout(), VK_IMAGE_LAYOUT_GENERAL, 0, 0,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, *batch._commandBuffer);
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

  std::vector<VkBufferImageCopy> copyRegions;
  copyRegions.reserve(bufferOffsets.size());
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

  vkCmdCopyBufferToImage(commandBuffer, stagingBuffer->getBuffer(), destination.getImage(), VK_IMAGE_LAYOUT_GENERAL,
                         static_cast<uint32_t>(copyRegions.size()), copyRegions.data());

  if (destination.getMipMapNumber() > 1) {
    _generateMipmaps(destination, commandBuffer);
  } else {
    const VkImageMemoryBarrier2 afterCopyBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
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
        .pImageMemoryBarriers = &afterCopyBarrier,
    };
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
  }
  batch._stagingBuffers.push_back(std::move(stagingBuffer));
}

UploadTicket ResourceUploader::submit() {
  if (_recordingBatch == nullptr) {
    throw std::logic_error("Can't submit an empty upload batch");
  }

  _recordingBatch->_commandBuffer->endCommands();
  const uint64_t completionValue = _nextCompletionValue++;
  const VkCommandBufferSubmitInfo commandBufferInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = _recordingBatch->_commandBuffer->getCommandBuffer(),
  };
  const VkSemaphoreSubmitInfo signalSemaphoreInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = _timelineSemaphore.getSemaphore(),
      .value = completionValue,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
  };
  const VkSubmitInfo2 submitInfo{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &commandBufferInfo,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &signalSemaphoreInfo,
  };

  const VkResult result = vkQueueSubmit2(_device->getQueue(QueueType::GRAPHICS), 1, &submitInfo, nullptr);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Resource upload submission failed: " + std::to_string(result));
  }

  _recordingBatch->_completionValue = completionValue;
  _inFlightBatches.push_back(std::move(_recordingBatch));
  return {
      .semaphore = _timelineSemaphore.getSemaphore(),
      .value = completionValue,
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
