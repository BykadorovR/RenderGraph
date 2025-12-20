module Swapchain;
import <limits>;
import <ranges>;
using namespace RenderGraph;

Swapchain::Swapchain(const MemoryAllocator& allocator, const Device& device)
    : _allocator(&allocator),
      _device(&device) {
  vkb::SwapchainBuilder builder{device.getDevice()};
  builder.set_composite_alpha_flags(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
  builder.set_desired_format(
      VkSurfaceFormatKHR{.format = _swapchainFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR});
  // because we use swapchain in compute shader
  builder.add_image_usage_flags(VK_IMAGE_USAGE_STORAGE_BIT);
  if (_verticalSync) {
    builder.set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR);
  } else {
    builder.set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR);
  }

  auto swapchainResult = builder.build();
  if (!swapchainResult) {
    throw std::runtime_error(swapchainResult.error().message());
  }
  // TODO: check extent
  _swapchain = swapchainResult.value();
}

void Swapchain::initialize() {
  // image views are created during this call, so need to assign to a variable
  auto images = _swapchain.get_images().value();
  auto imageViews = _swapchain.get_image_views().value();
  for (auto&& [image, imageView] : std::views::zip(images, imageViews)) {
    auto imageWrap = std::make_unique<Image>(*_allocator);
    imageWrap->wrapImage(image, _swapchainFormat, {_swapchain.extent.width, _swapchain.extent.height}, 1, 1);    
    auto imageViewWrap = std::make_unique<ImageView>(*_device);
    imageViewWrap->wrapImageView(imageView, std::move(imageWrap));
    _imageViews.push_back(std::move(imageViewWrap));
  }
}

VkResult Swapchain::acquireNextImage(const Semaphore& semaphore) noexcept {
  // RETURNS ONLY INDEX, NOT IMAGE
  // semaphore to signal, once image is available
  return vkAcquireNextImageKHR(_device->getLogicalDevice(), _swapchain,
                                      std::numeric_limits<std::uint64_t>::max(), semaphore.getSemaphore(),
                                      nullptr, &_swapchainIndex);
}

Swapchain::~Swapchain() { _destroy(); }

void Swapchain::_destroy() {
  _imageViews.clear();
  vkb::destroy_swapchain(_swapchain);
}

Image& Swapchain::getImage(int index) const noexcept { return _imageViews[index]->getImage(); }

const ImageView& Swapchain::getImageView(int index) const noexcept { return *_imageViews.at(index); }

std::vector<std::shared_ptr<ImageView>> Swapchain::getImageViews() const noexcept { return _imageViews; };

int Swapchain::getImageCount() const noexcept { return _imageViews.size(); }

const vkb::Swapchain& Swapchain::getSwapchain() const noexcept { return _swapchain; }

uint32_t Swapchain::getSwapchainIndex() const noexcept { return _swapchainIndex; }

std::vector<std::shared_ptr<ImageView>> Swapchain::reset() {
  if (vkDeviceWaitIdle(_device->getDevice().device) != VK_SUCCESS)
    throw std::runtime_error("failed to create reset swap chain!");

  vkb::SwapchainBuilder builder{_device->getDevice()};
  builder.set_old_swapchain(_swapchain);
  builder.set_composite_alpha_flags(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
  builder.set_desired_format(
      VkSurfaceFormatKHR{.format = _swapchainFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR});
  // because we use swapchain in compute shader
  builder.add_image_usage_flags(VK_IMAGE_USAGE_STORAGE_BIT);
  auto swapchainResult = builder.build();
  if (!swapchainResult) {
    // If it failed to create a swapchain, the old swapchain handle is invalid.
    _swapchain.swapchain = nullptr;
  }
  auto imageViewsOld = _imageViews;

  // Even though we recycled the previous swapchain, we need to free its resources.
  _destroy();
  // Get the new swapchain and place it in our variable
  _swapchain = swapchainResult.value();

  initialize();

  return imageViewsOld;
}