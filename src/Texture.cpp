module;

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <algorithm>
#include <functional>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vk_mem_alloc.h>
#include <volk.h>

module Texture;

using namespace RenderGraph;

Image::Image(const MemoryAllocator& memoryAllocator) : _memoryAllocator(&memoryAllocator) {}

void Image::createImage(VkFormat format,
                        glm::ivec2 resolution,
                        int mipMapNumber,
                        int layerNumber,
                        VkImageAspectFlags aspectMask,
                        VkImageUsageFlags usage) {
  _format = format;
  _resolution = resolution;
  _mipMapNumber = mipMapNumber;
  _layerNumber = layerNumber;
  _aspectMask = aspectMask;
  _usageFlags = usage;
  _imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkImageCreateInfo imageInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                              .imageType = VK_IMAGE_TYPE_2D,
                              .format = format,
                              .extent = {.width = static_cast<uint32_t>(resolution.x),
                                         .height = static_cast<uint32_t>(resolution.y),
                                         .depth = 1},
                              .mipLevels = static_cast<uint32_t>(mipMapNumber),
                              .arrayLayers = static_cast<uint32_t>(layerNumber),
                              .samples = VK_SAMPLE_COUNT_1_BIT,
                              .tiling = VK_IMAGE_TILING_OPTIMAL,
                              .usage = usage,
                              .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                              .initialLayout = _imageLayout};

  VmaAllocationCreateInfo allocCreateInfo = {};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

  auto sts = vmaCreateImage(_memoryAllocator->getAllocator(), &imageInfo, &allocCreateInfo, &_image, &_imageMemory,
                            nullptr);
  if (sts != VK_SUCCESS) throw std::invalid_argument("Can't create an image " + std::to_string(sts));
}

void Image::wrapImage(const VkImage& existingImage,
                      VkFormat format,
                      glm::ivec2 resolution,
                      int mipMapNumber,
                      int layerNumber,
                      VkImageAspectFlags aspectMask,
                      VkImageUsageFlags usage) {
  _image = existingImage;
  _format = format;
  _resolution = resolution;
  _mipMapNumber = mipMapNumber;
  _layerNumber = layerNumber;
  _aspectMask = aspectMask;
  _usageFlags = usage;
}

VkImageAspectFlags Image::getAspectMask() const noexcept { return _aspectMask; }

VkImageUsageFlags Image::getUsageFlags() const noexcept { return _usageFlags; }

void Image::changeLayout(VkImageLayout oldLayout,
                         VkImageLayout newLayout,
                         VkPipelineStageFlags2 srcStageMask,
                         VkAccessFlags2 srcAccessMask,
                         VkPipelineStageFlags2 dstStageMask,
                         VkAccessFlags2 dstAccessMask,
                         const CommandBuffer& commandBuffer) {
  _imageLayout = newLayout;
  VkImageMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .srcStageMask = srcStageMask,
                                .srcAccessMask = srcAccessMask,
                                .dstStageMask = dstStageMask,
                                .dstAccessMask = dstAccessMask,
                                .oldLayout = oldLayout,
                                .newLayout = newLayout,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = _image,
                                .subresourceRange = {.aspectMask = _aspectMask,
                                                     .baseMipLevel = 0,
                                                     .levelCount = static_cast<uint32_t>(_mipMapNumber),
                                                     .baseArrayLayer = 0,
                                                     .layerCount = static_cast<uint32_t>(_layerNumber)}};
  VkDependencyInfo dependencyInfo{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                  .imageMemoryBarrierCount = 1,
                                  .pImageMemoryBarriers = &barrier};

  vkCmdPipelineBarrier2(commandBuffer.getCommandBuffer(), &dependencyInfo);
}

glm::ivec2 Image::getResolution() const noexcept { return _resolution; }

VkImage Image::getImage() const noexcept { return _image; }

VkFormat Image::getFormat() const noexcept { return _format; }

VkImageLayout Image::getImageLayout() const noexcept { return _imageLayout; }

int Image::getMipMapNumber() const noexcept { return _mipMapNumber; }

int Image::getLayerNumber() const noexcept { return _layerNumber; }

void Image::destroy() {
  if (_imageMemory) {
    vmaDestroyImage(_memoryAllocator->getAllocator(), _image, _imageMemory);
    _image = {};
    _imageMemory = nullptr;
  }
}

Image::~Image() { destroy(); }

ImageView::ImageView(std::unique_ptr<Image> image, const Device& device) noexcept
    : _image(std::move(image)),
      _device(&device) {}

void ImageView::createImageView(VkImageViewType type, int baseMipMap, int baseArrayLayer) {
  _type = type;
  _baseMipMap = baseMipMap;
  _baseArrayLayer = baseArrayLayer;

  VkImageViewCreateInfo viewInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                 .image = _image->getImage(),
                                 .viewType = type,
                                 .format = _image->getFormat(),
                                 .components = VkComponentMapping{.r = VK_COMPONENT_SWIZZLE_IDENTITY,
                                                                  .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                                                                  .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                                                                  .a = VK_COMPONENT_SWIZZLE_IDENTITY},
                                 .subresourceRange = {
                                     .aspectMask = _image->getAspectMask(),
                                     .baseMipLevel = static_cast<uint32_t>(baseMipMap),
                                     .levelCount = static_cast<uint32_t>(_image->getMipMapNumber()),
                                     .baseArrayLayer = static_cast<uint32_t>(baseArrayLayer),
                                     .layerCount = static_cast<uint32_t>(_image->getLayerNumber()),
                                 }};

  if (vkCreateImageView(_device->getLogicalDevice(), &viewInfo, nullptr, &_imageView) != VK_SUCCESS) {
    throw std::runtime_error("failed to create texture image view!");
  }
}
void ImageView::wrapImageView(const VkImageView& imageView) { _imageView = imageView; }

VkImageView ImageView::getImageView() const noexcept { return _imageView; }

Image& ImageView::getImage() const noexcept { return *_image; }

VkImageViewType ImageView::getType() const noexcept { return _type; }

int ImageView::getBaseMipMap() const noexcept { return _baseMipMap; }

int ImageView::getBaseArrayLayer() const noexcept { return _baseArrayLayer; }

void ImageView::destroy() {
  if (_imageView) {
    vkDestroyImageView(_device->getLogicalDevice(), _imageView, nullptr);
    _imageView = {};
  }
}

ImageView::~ImageView() { destroy(); }

ImageViewHolder::ImageViewHolder(std::vector<std::shared_ptr<ImageView>> imageViews,
                                 std::function<int()> index) noexcept {
  _imageViews = imageViews;
  _index = index;
}

void ImageViewHolder::setImageViews(std::vector<std::shared_ptr<ImageView>> imageViews) { _imageViews = imageViews; }

const ImageView& ImageViewHolder::getImageView() const { return *_imageViews[_index()]; }

std::function<int()> ImageViewHolder::getIndexFunction() const { return _index; }

int ImageViewHolder::getIndex() const { return _index(); }

std::vector<ImageView*> ImageViewHolder::getImageViews() const {
  return _imageViews | std::views::transform([](auto const& iv) { return iv.get(); }) |
         std::ranges::to<std::vector<ImageView*>>();
}

bool ImageViewHolder::contains(const std::vector<std::shared_ptr<ImageView>>& imageViews) const noexcept {
  return std::ranges::equal(imageViews, _imageViews, [](const auto& lhs, const auto& rhs) {
    return lhs->getImage().getImage() == rhs->getImage().getImage();
  });
}

Sampler::Sampler(const Device& device) noexcept : _device(&device) {}

void Sampler::createSampler(VkSamplerAddressMode mode, int mipMapLevels, int anisotropicSamples, VkFilter filter) {
  VkSamplerCreateInfo samplerInfo{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = filter,
      .minFilter = filter,
      .addressModeU = mode,
      .addressModeV = mode,
      .addressModeW = mode,
      .mipLodBias = 0.0f,
      .anisotropyEnable = anisotropicSamples > 0 ? true : false,
      .maxAnisotropy = std::min(_device->getDevice().physical_device.properties.limits.maxSamplerAnisotropy,
                                static_cast<float>(anisotropicSamples)),
      .compareEnable = false,
      .minLod = 0.0f,
      .maxLod = static_cast<float>(mipMapLevels),
      .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
      .unnormalizedCoordinates = false};
  if (mipMapLevels > 1) samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  if (vkCreateSampler(_device->getLogicalDevice(), &samplerInfo, nullptr, &_sampler) != VK_SUCCESS) {
    throw std::runtime_error("failed to create texture sampler!");
  }
}

VkSampler Sampler::getSampler() const noexcept { return _sampler; }

Sampler::~Sampler() { vkDestroySampler(_device->getLogicalDevice(), _sampler, nullptr); }

Texture::Texture(std::shared_ptr<ImageView> imageView, std::shared_ptr<Sampler> sampler) noexcept {
  _imageView = imageView;
  _sampler = sampler;
}

const ImageView& Texture::getImageView() const noexcept { return *_imageView; }

const Sampler* Texture::getSampler() const noexcept { return _sampler.get(); }
