module;
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include "vk_mem_alloc.h"
module Buffer;
using namespace RenderGraph;

Buffer::Buffer(VkDeviceSize size,
               VkBufferUsageFlags usage,
               VmaAllocationCreateFlags flags,
               const MemoryAllocator& memoryAllocator)
    : _memoryAllocator(&memoryAllocator) {
  _size = size;

  VkBufferCreateInfo bufferInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = size,
                                .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  VmaAllocationCreateInfo allocCreateInfo = {.flags = flags, .usage = VMA_MEMORY_USAGE_AUTO};

  auto result = vmaCreateBuffer(_memoryAllocator->getAllocator(), &bufferInfo, &allocCreateInfo, &_buffer, &_allocation,
                                &_allocationInfo);
  if (result != VK_SUCCESS) throw std::runtime_error("Can't vmaCreateBuffer " + result);
}

/*
 * For resources that you frequently write on CPU via mapped pointer and frequently read on GPU e.g. as a uniform
 * buffer (also called "dynamic"), multiple options are possible: Easiest solution is to have one copy of the resource
 * in HOST_VISIBLE memory, even if it means system RAM (not DEVICE_LOCAL) on systems with a discrete graphics card,
 * and make the device reach out to that resource directly. Reads performed by the device will then go through PCI
 * Express bus. The performance of this access may be limited, but it may be fine depending on the size of this
 * resource (whether it is small enough to quickly end up in GPU cache) and the sparsity of access. On systems with
 * unified memory (e.g. AMD APU or Intel integrated graphics, mobile chips), a memory type may be available that is
 * both HOST_VISIBLE (available for mapping) and DEVICE_LOCAL (fast to access from the GPU). Then, it is likely the
 * best choice for such type of resource. Systems with a discrete graphics card and separate video memory may or may
 * not expose a memory type that is both HOST_VISIBLE and DEVICE_LOCAL, also known as Base Address Register (BAR). If
 * they do, it represents a piece of VRAM (or entire VRAM, if ReBAR is enabled in the motherboard BIOS) that is
 * available to CPU for mapping. Writes performed by the host to that memory go through PCI Express bus. The
 * performance of these writes may be limited, but it may be fine, especially on PCIe 4.0, as long as rules of using
 * uncached and write-combined memory are followed - only sequential writes and no reads. Finally, you may need or
 * prefer to create a separate copy of the resource in DEVICE_LOCAL memory, a separate "staging" copy in HOST_VISIBLE
 * memory and perform an explicit transfer command between them. Thankfully, VMA offers an aid to create and use such
 * resources in the the way optimal for the current Vulkan device. To help the library make the best choice, use flag
 * VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT together with
 * VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT. It will then prefer a memory type that is both
 * DEVICE_LOCAL and HOST_VISIBLE (integrated memory or BAR), but if no such memory type is available or allocation
 * from it fails (PC graphics cards have only 256 MB of BAR by default, unless ReBAR is supported and enabled in
 * BIOS), it will fall back to DEVICE_LOCAL memory for fast GPU access. It is then up to you to detect that the
 * allocation ended up in a memory type that is not HOST_VISIBLE, so you need to create another "staging" allocation
 * and perform explicit transfers.
 */
void Buffer::setData(std::span<const std::byte> data, const CommandBuffer& commandBufferTransfer) {
  VkMemoryPropertyFlags memPropFlags;
  vmaGetAllocationMemoryProperties(_memoryAllocator->getAllocator(), _allocation, &memPropFlags);

  // The Allocation ended up in a mappable memory.
  if (memPropFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    // Calling vmaCopyMemoryToAllocation() does vmaMapMemory(), memcpy(), vmaUnmapMemory(), and vmaFlushAllocation().
    auto result = vmaCopyMemoryToAllocation(_memoryAllocator->getAllocator(), data.data(), _allocation, 0, data.size());
    if (result != VK_SUCCESS) throw std::runtime_error("Can't vmaCopyMemoryToAllocation " + result);

    VkBufferMemoryBarrier barrierCopy = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                         .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
                                         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                                         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                         .buffer = _buffer,
                                         .offset = 0,
                                         .size = data.size()};

    // It's important to insert a buffer memory barrier here to ensure writing to the buffer has finished.
    vkCmdPipelineBarrier(commandBufferTransfer.getCommandBuffer(), VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_ALL_SHADER_BITS, 0, 0, nullptr, 1, &barrierCopy, 0, nullptr);
  } else {
    _bufferStaging = std::make_unique<Buffer>(
        data.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, *_memoryAllocator);

    // Calling vmaCopyMemoryToAllocation() does vmaMapMemory(), memcpy(), vmaUnmapMemory(), and vmaFlushAllocation().
    auto result = vmaCopyMemoryToAllocation(_memoryAllocator->getAllocator(), data.data(),
                                            _bufferStaging->getAllocation(), 0, data.size());
    if (result != VK_SUCCESS) throw std::runtime_error("Can't vmaCopyMemoryToAllocation " + result);

    VkBufferMemoryBarrier barrierStage = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrierStage.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrierStage.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrierStage.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierStage.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierStage.buffer = _bufferStaging->getBuffer();
    barrierStage.offset = 0;
    barrierStage.size = data.size();

    // Insert a buffer memory barrier to make sure writing to the staging buffer has finished.
    vkCmdPipelineBarrier(commandBufferTransfer.getCommandBuffer(), VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrierStage, 0, nullptr);

    // copy from staging buffer to current buffer
    VkBufferCopy copyRegion{.srcOffset = 0, .dstOffset = 0, .size = data.size()};
    vkCmdCopyBuffer(commandBufferTransfer.getCommandBuffer(), _bufferStaging->getBuffer(), _buffer, 1, &copyRegion);

    VkBufferMemoryBarrier barrierCopy = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                                         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                         .buffer = _buffer,
                                         .offset = 0,
                                         .size = data.size()};

    // Make sure copying from staging buffer to the actual buffer has finished by inserting a buffer memory barrier.
    vkCmdPipelineBarrier(commandBufferTransfer.getCommandBuffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_SHADER_BITS, 0, 0, nullptr, 1, &barrierCopy, 0, nullptr);
  }
}

VkDeviceSize Buffer::getSize() const noexcept { return _size; }

const VmaAllocationInfo& Buffer::getAllocationInfo() const noexcept { return _allocationInfo; }

VmaAllocation Buffer::getAllocation() const noexcept { return _allocation; }

VkBuffer Buffer::getBuffer() const noexcept { return _buffer; }

Buffer::~Buffer() { vmaDestroyBuffer(_memoryAllocator->getAllocator(), _buffer, _allocation); }