module;

#include <VkBootstrap.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>
#include <volk.h>

module Swapchain;

using namespace RenderGraph;

vkb::SwapchainBuilder Swapchain::_createBuilder() const {
  const auto& device = _device->getDevice();
  if (device.surface == VK_NULL_HANDLE) throw std::runtime_error("Swapchain requires a surface");
  const auto graphicsFamily = static_cast<uint32_t>(_device->getQueueIndex(QueueType::GRAPHICS));
  auto sharingFamily = static_cast<uint32_t>(_device->getQueueIndex(QueueType::COMPUTE));
  if (sharingFamily == graphicsFamily) {
    // Still select CONCURRENT when another device family exists (e.g. transfer).
    const auto& families = _device->getResourceQueueFamilies();
    sharingFamily = families.front() != graphicsFamily ? families.front() : families.back();
  }
  // vk-bootstrap uses these two indices for sharing. Presentation uses graphics;
  // include compute so both graph queues can access the swapchain images.
  vkb::SwapchainBuilder builder{_device->getPhysicalDevice(), _device->getLogicalDevice(), device.surface,
                                sharingFamily, graphicsFamily};
  builder.set_allocation_callbacks(device.allocation_callbacks);
  return builder;
}

Swapchain::Swapchain(glm::ivec2 resolution,
                     const MemoryAllocator& allocator,
                     const Device& device,
                     VkImageUsageFlags imageUsage)
    : _allocator(&allocator),
      _device(&device) {
  auto builder = _createBuilder();
#if defined(__ANDROID__)
  builder.set_pre_transform_flags(VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR);
#endif
  builder.set_desired_extent(static_cast<uint32_t>(resolution.x), static_cast<uint32_t>(resolution.y));
  builder.set_composite_alpha_flags(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
  builder.set_desired_format(
      VkSurfaceFormatKHR{.format = _swapchainFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR});
  if (imageUsage & VK_IMAGE_USAGE_STORAGE_BIT) {
    // The default sRGB fallback cannot be used as a storage image.
    builder.add_fallback_format({VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR});
  }
  builder.set_image_usage_flags(imageUsage);
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
  // The raw-handle builder does not inherit the device's instance version.
  _swapchain.instance_version = device.getDevice().instance_version;
  _swapchainFormat = _swapchain.image_format;
}

void Swapchain::initialize() {
  // image views are created during this call, so need to assign to a variable
  auto images = _swapchain.get_images().value();
  auto imageViews = _swapchain.get_image_views().value();
  for (auto&& [image, imageView] : std::views::zip(images, imageViews)) {
    auto imageWrap = std::make_unique<Image>(*_allocator, *_device);
    imageWrap->wrapImage(image, _swapchainFormat, {_swapchain.extent.width, _swapchain.extent.height}, 1, 1,
                         VK_IMAGE_ASPECT_COLOR_BIT, _swapchain.image_usage_flags);
    auto imageViewWrap = std::make_unique<ImageView>(std::move(imageWrap), *_device);
    imageViewWrap->wrapImageView(imageView);
    _imageViews.push_back(std::move(imageViewWrap));
  }
}

VkResult Swapchain::acquireNextImage(const Semaphore& semaphore) noexcept {
  // RETURNS ONLY INDEX, NOT IMAGE
  // semaphore to signal, once image is available
  return vkAcquireNextImageKHR(_device->getLogicalDevice(), _swapchain, std::numeric_limits<std::uint64_t>::max(),
                               semaphore.getSemaphore(), nullptr, &_swapchainIndex);
}

Swapchain::~Swapchain() { _destroy(); }

void Swapchain::_destroy() {
  _imageViews.clear();
  _imageViews.shrink_to_fit();
  vkb::destroy_swapchain(_swapchain);
  _swapchain.swapchain = VK_NULL_HANDLE;
}

Image& Swapchain::getImage(int index) const noexcept { return _imageViews[index]->getImage(); }

std::vector<std::shared_ptr<ImageView>> Swapchain::getImageViews() const { return _imageViews; };

int Swapchain::getImageCount() const noexcept { return _imageViews.size(); }

const vkb::Swapchain& Swapchain::getSwapchain() const noexcept { return _swapchain; }

uint32_t Swapchain::getSwapchainIndex() const noexcept { return _swapchainIndex; }

std::vector<std::shared_ptr<ImageView>> Swapchain::reset(glm::ivec2 resolution) {
  auto builder = _createBuilder();
#if defined(__ANDROID__)
  builder.set_pre_transform_flags(VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR);
#endif
  builder.set_desired_extent(static_cast<uint32_t>(resolution.x), static_cast<uint32_t>(resolution.y));
  builder.set_old_swapchain(_swapchain);
  builder.set_composite_alpha_flags(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
  builder.set_desired_format(
      VkSurfaceFormatKHR{.format = _swapchainFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR});
  builder.set_image_usage_flags(_swapchain.image_usage_flags);
  auto swapchainResult = builder.build();
  if (!swapchainResult) {
    // Passing oldSwapchain can retire it even when creation fails.
    _destroy();
    throw std::runtime_error(swapchainResult.error().message());
  }
  auto imageViewsOld = _imageViews;

  // Even though we recycled the previous swapchain, we need to free its resources.
  _destroy();
  // Get the new swapchain and place it in our variable
  _swapchain = swapchainResult.value();
  _swapchain.instance_version = _device->getDevice().instance_version;
  _swapchainFormat = _swapchain.image_format;

  initialize();

  return imageViewsOld;
}
