module;

#include <VkBootstrap.h>
#include <memory>
#include <vector>

export module Swapchain;

import Allocator;
import Device;
import Texture;
import Command;
import Sync;
import glm;

export namespace RenderGraph {
class Swapchain {
 private:
  const MemoryAllocator* _allocator;
  const Device* _device;
  vkb::Swapchain _swapchain;
  uint32_t _swapchainIndex = 0;
  VkFormat _swapchainFormat = VK_FORMAT_R8G8B8A8_UNORM;
  std::vector<std::shared_ptr<ImageView>> _imageViews;
  bool _verticalSync = false;
  void _destroy();

 public:
  Swapchain(glm::ivec2 resolution,
            const MemoryAllocator& allocator,
            const Device& device,
            VkImageUsageFlags imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
  Swapchain(const Swapchain&) = delete;
  Swapchain& operator=(const Swapchain&) = delete;
  Swapchain(Swapchain&&) = delete;
  Swapchain& operator=(Swapchain&&) = delete;

  void initialize();
  VkResult acquireNextImage(const Semaphore& semaphore) noexcept;
  std::vector<std::shared_ptr<ImageView>> reset(glm::ivec2 resolution);

  // to be able change layout
  Image& getImage(int index) const noexcept;
  std::vector<std::shared_ptr<ImageView>> getImageViews() const;
  int getImageCount() const noexcept;
  const vkb::Swapchain& getSwapchain() const noexcept;
  uint32_t getSwapchainIndex() const noexcept;
  ~Swapchain();
};
}  // namespace RenderGraph
