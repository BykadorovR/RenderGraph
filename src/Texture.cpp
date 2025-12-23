module;
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
module Texture;
import <volk.h>;
import <algorithm>;
import <ranges>;
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

  VkImageCreateInfo imageInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
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
      .initialLayout = _imageLayout,
  };

  VmaAllocationCreateInfo allocCreateInfo = {};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

  auto sts = vmaCreateImage(_memoryAllocator->getAllocator(), &imageInfo, &allocCreateInfo, &_image, &_imageMemory,
                            nullptr);
  if (sts != VK_SUCCESS) throw std::invalid_argument("Can't create an image " + sts);
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

void Image::copyFrom(std::unique_ptr<Buffer> buffer,
                     const std::vector<int>& bufferOffsets,
                     const CommandBuffer& commandBuffer) {
  _stagingBuffer = std::move(buffer);

  std::vector<VkBufferImageCopy> bufferCopyRegions;
  bufferCopyRegions.reserve(bufferOffsets.size());
  for (int i = 0; i < bufferOffsets.size(); i++) {
    VkBufferImageCopy region{
        .bufferOffset = static_cast<VkDeviceSize>(bufferOffsets[i]),
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {.aspectMask = _aspectMask,
                             .mipLevel = 0,
                             .baseArrayLayer = static_cast<uint32_t>(i),
                             .layerCount = 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {static_cast<uint32_t>(_resolution.x), static_cast<uint32_t>(_resolution.y), 1}};

    bufferCopyRegions.push_back(region);
  }

  // demands image to be in VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL before copy
  vkCmdCopyBufferToImage(commandBuffer.getCommandBuffer(), _stagingBuffer->getBuffer(), _image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, bufferCopyRegions.size(), bufferCopyRegions.data());
  // need to insert memory barrier so read in fragment shader waits for copy
  VkMemoryBarrier memoryBarrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                   .pNext = nullptr,
                                   .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                   .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
  vkCmdPipelineBarrier(commandBuffer.getCommandBuffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}

VkImageAspectFlags Image::getAspectMask() const noexcept { return _aspectMask; }

VkImageUsageFlags Image::getUsageFlags() const noexcept { return _usageFlags; }

void Image::changeLayout(VkImageLayout oldLayout,
                         VkImageLayout newLayout,
                         VkAccessFlags srcAccessMask,
                         VkAccessFlags dstAccessMask,
                         const CommandBuffer& commandBuffer) {
  _imageLayout = newLayout;
  VkImageMemoryBarrier barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                               .srcAccessMask = srcAccessMask,
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

  vkCmdPipelineBarrier(commandBuffer.getCommandBuffer(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void Image::overrideLayout(VkImageLayout layout) { _imageLayout = layout; }

void Image::generateMipmaps(const CommandBuffer& commandBuffer) {
  VkImageMemoryBarrier barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .image = getImage(),
                               .subresourceRange = {.aspectMask = _aspectMask,
                                                    .levelCount = 1,
                                                    .baseArrayLayer = 0,
                                                    .layerCount = static_cast<uint32_t>(_layerNumber)}};
  auto mipResolution = getResolution();
  int mipWidth = mipResolution.x, mipHeight = mipResolution.y;
  for (uint32_t i = 1; i < _mipMapNumber; i++) {
    // change layout of source image to SRC so we can resize it and generate i-th mip map level
    barrier.subresourceRange.baseMipLevel = i - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer.getCommandBuffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkImageBlit blit{};
    blit.srcOffsets[0] = {0, 0, 0};
    blit.srcOffsets[1] = {mipWidth, mipHeight, 1};  // previous image size
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.mipLevel = i - 1;  // previous mip map image used for resize
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = _layerNumber;
    blit.dstOffsets[0] = {0, 0, 0};
    blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.mipLevel = i;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = _layerNumber;

    // i = 0 has SRC layout (we changed it above), i = 1 has DST layout (we changed from undefined to dst in
    // constructor)
    vkCmdBlitImage(commandBuffer.getCommandBuffer(), getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, getImage(),
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

    // change i = 0 to READ OPTIMAL, we won't use this level anymore, next resizes will use next i
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer.getCommandBuffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    if (mipWidth > 1) mipWidth /= 2;
    if (mipHeight > 1) mipHeight /= 2;
  }

  // we don't generate mip map from last level so we need explicitly change dst to read only
  barrier.subresourceRange.baseMipLevel = _mipMapNumber - 1;
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(commandBuffer.getCommandBuffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

  // we changed real image layout above, need to override imageLayout internal field
  overrideLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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

void ImageView::destroy() { vkDestroyImageView(_device->getLogicalDevice(), _imageView, nullptr); }

ImageView::~ImageView() { destroy(); }

ImageViewHolder::ImageViewHolder(std::vector<std::shared_ptr<ImageView>> imageViews,
                                 std::function<int()> index) noexcept {
  _imageViews = imageViews;
  _index = index;
}

void ImageViewHolder::setReset(bool reset) noexcept { _reset = reset; }

bool ImageViewHolder::getReset() const noexcept { return _reset; }

const ImageView& ImageViewHolder::getImageView() const noexcept { return *_imageViews[_index()]; }

std::function<int()> ImageViewHolder::getIndexFunction() const noexcept { return _index; }

int ImageViewHolder::getIndex() const noexcept { return _index(); }

std::vector<ImageView*> ImageViewHolder::getImageViews() const noexcept {
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
  VkSamplerCreateInfo samplerInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                  .magFilter = filter,
                                  .minFilter = filter,
                                  .addressModeU = mode,
                                  .addressModeV = mode,
                                  .addressModeW = mode,
                                  .mipLodBias = 0.0f,
                                  .anisotropyEnable = anisotropicSamples > 0 ? true : false,
                                  .maxAnisotropy = std::min(_device->getDeviceProperties().limits.maxSamplerAnisotropy,
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

const Sampler& Texture::getSampler() const noexcept { return *_sampler; }