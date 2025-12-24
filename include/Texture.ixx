module;
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
export module Texture;
import <vk_mem_alloc.h>;
import Buffer;
import Allocator;
import Command;
import Device;
import glm;
import <volk.h>;
import <memory>;
import <vector>;
import <functional>;
import <stdexcept>;

export namespace RenderGraph {
template <class T>
concept Arithmetic = std::integral<T> || std::floating_point<T>;

template <Arithmetic T>
class ImageCPU final {
 private:
  T* _data;
  std::function<void(T*)> _deleter;
  glm::ivec2 _resolution;
  int _channels;

 public:
  void setData(T* data, std::function<void(T*)> deleter) {
    _data = data;
    _deleter = deleter;
  }
  void setResolution(glm::ivec2 resolution) { _resolution = resolution; }
  void setChannels(int channels) { _channels = channels; }

  const T* getData() const noexcept { return _data; }
  glm::ivec2 getResolution() const noexcept { return _resolution; }
  int getChannels() const noexcept { return _channels; }
  ~ImageCPU() { _deleter(_data); }
};

class Image final {
 private:
  const MemoryAllocator* _memoryAllocator;
  VkImage _image;
  VmaAllocation _imageMemory = nullptr;
  std::unique_ptr<Buffer> _stagingBuffer;
  // image mandatory options
  VkFormat _format;
  glm::ivec2 _resolution;
  int _mipMapNumber = 1;
  int _layerNumber = 1;
  VkImageAspectFlags _aspectMask;
  VkImageUsageFlags _usageFlags;
  VkImageLayout _imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

 public:
  Image(const MemoryAllocator& memoryAllocator);
  Image(const Image&) = delete;
  Image& operator=(const Image&) = delete;
  Image(Image&&) = delete;
  Image& operator=(Image&&) = delete;

  void createImage(VkFormat format,
                   glm::ivec2 resolution,
                   int mipMapNumber,
                   int layerNumber,
                   VkImageAspectFlags aspectMask,
                   VkImageUsageFlags usage);
  void wrapImage(const VkImage& existingImage,
                 VkFormat format,
                 glm::ivec2 resolution,
                 int mipMapNumber,
                 int layerNumber,
                 VkImageAspectFlags aspectMask,
                 VkImageUsageFlags usage);

  // bufferOffsets contains offsets for part of buffer that should be copied to corresponding layers of image
  void copyFrom(std::unique_ptr<Buffer> buffer,
                const std::vector<int>& bufferOffsets,
                const CommandBuffer& commandBuffer);
  void changeLayout(VkImageLayout oldLayout,
                    VkImageLayout newLayout,
                    VkAccessFlags srcAccessMask,
                    VkAccessFlags dstAccessMask,
                    const CommandBuffer& commandBuffer);
  void overrideLayout(VkImageLayout layout);
  void generateMipmaps(const CommandBuffer& commandBuffer);

  glm::ivec2 getResolution() const noexcept;
  VkImage getImage() const noexcept;
  VkFormat getFormat() const noexcept;
  VkImageLayout getImageLayout() const noexcept;
  int getMipMapNumber() const noexcept;
  int getLayerNumber() const noexcept;
  VkImageAspectFlags getAspectMask() const noexcept;
  VkImageUsageFlags getUsageFlags() const noexcept;
  void destroy();

  ~Image();
};

class ImageView final {
 private:
  const Device* _device;
  std::unique_ptr<Image> _image;
  VkImageView _imageView;
  VkImageViewType _type;
  int _baseMipMap, _baseArrayLayer;

 public:
  ImageView(std::unique_ptr<Image> image, const Device& device) noexcept;
  ImageView(const ImageView&) = delete;
  ImageView& operator=(const ImageView&) = delete;
  ImageView(ImageView&&) = delete;
  ImageView& operator=(ImageView&&) = delete;

  // componentMapping to pass BGRA texture to shader that accepts only RGBA
  // baseArrayLayer - which layer/face is used
  // baseMipMapLevel - which mip map level is used
  void createImageView(VkImageViewType type,
                       int baseMipMap,
                       int baseArrayLayer);
  void wrapImageView(const VkImageView& imageView);
  VkImageView getImageView() const noexcept;
  Image& getImage() const noexcept;
  VkImageViewType getType() const noexcept;
  int getBaseMipMap() const noexcept;
  int getBaseArrayLayer() const noexcept;
  void destroy();
  ~ImageView();
};

class ImageViewHolder final {
 protected:
  std::vector<std::shared_ptr<ImageView>> _imageViews;
  std::function<int()> _index;
 public:
  ImageViewHolder(std::vector<std::shared_ptr<ImageView>> imageViews, std::function<int()> index) noexcept;
  ImageViewHolder(const ImageViewHolder&) = delete;
  ImageViewHolder& operator=(const ImageViewHolder&) = delete;
  ImageViewHolder(ImageViewHolder&&) = delete;
  ImageViewHolder& operator=(ImageViewHolder&&) = delete;

  void setImageViews(std::vector<std::shared_ptr<ImageView>> imageViews);
  const ImageView& getImageView() const noexcept;
  std::function<int()> getIndexFunction() const noexcept;
  int getIndex() const noexcept;
  std::vector<ImageView*> getImageViews() const noexcept;
  bool contains(const std::vector<std::shared_ptr<ImageView>>& imageViews) const noexcept;
};

class Sampler final {
 private:
  const Device* _device;
  VkSampler _sampler;

 public:
  Sampler(const Device& device) noexcept;
  Sampler(const Sampler&) = delete;
  Sampler& operator=(const Sampler&) = delete;
  Sampler(Sampler&&) = delete;
  Sampler& operator=(Sampler&&) = delete;

  void createSampler(VkSamplerAddressMode mode, int mipMapLevels, int anisotropicSamples, VkFilter filter);
  VkSampler getSampler() const noexcept;
  ~Sampler();
};

class Texture final {
 private:
  std::shared_ptr<ImageView> _imageView;
  std::shared_ptr<Sampler> _sampler;

 public:
  Texture(std::shared_ptr<ImageView> imageView, std::shared_ptr<Sampler> sampler) noexcept;
  Texture(const Texture&) = delete;
  Texture& operator=(const Texture&) = delete;
  Texture(Texture&&) = delete;
  Texture& operator=(Texture&&) = delete;

  const ImageView& getImageView() const noexcept;
  const Sampler& getSampler() const noexcept;
};
}  // namespace RenderGraph