export module Swapchain;
import Allocator;
import Device;
import Texture;
import Command;
import Sync;
import glm;
import <volk.h>;
import <VkBootstrap.h>;
import <memory>;

export namespace RenderGraph{class Swapchain {
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
  Swapchain(const MemoryAllocator& allocator, const Device& device);
  Swapchain(const Swapchain&) = delete;
  Swapchain& operator=(const Swapchain&) = delete;
  Swapchain(Swapchain&&) = delete;
  Swapchain& operator=(Swapchain&&) = delete;

  void initialize();
  VkResult acquireNextImage(const Semaphore& semaphore) noexcept;
  std::vector<std::shared_ptr<ImageView>> reset();

  // to be able change layout
  Image& getImage(int index) const noexcept;
  const ImageView& getImageView(int index) const noexcept;
  std::vector<std::shared_ptr<ImageView>> getImageViews() const noexcept;
  int getImageCount() const noexcept;
  const vkb::Swapchain& getSwapchain() const noexcept;
  uint32_t getSwapchainIndex() const noexcept;
  ~Swapchain();
};
}  // namespace RenderGraph