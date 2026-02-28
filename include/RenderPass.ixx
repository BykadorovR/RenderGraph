export module RenderPass;
import Device;
import <volk.h>;
import <vector>;
import <optional>;
import <stdexcept>;

export namespace RenderGraph {
class RenderPass final {
 private:
  const Device* _device;
  VkRenderPass _renderPass = VK_NULL_HANDLE;
  std::vector<VkFramebuffer> _framebuffers;

  void _createFramebuffers(const std::vector<std::vector<VkImageView>>& colorImageViews,
                            const std::vector<VkImageView>& depthImageViews,
                            VkExtent2D extent);
  void _destroyFramebuffers();

 public:
  RenderPass(const Device& device) noexcept;
  RenderPass(const RenderPass&) = delete;
  RenderPass& operator=(const RenderPass&) = delete;
  RenderPass(RenderPass&&) = delete;
  RenderPass& operator=(RenderPass&&) = delete;

  // colorImageViews[attachment][framebuffer] - one framebuffer per swapchain image or frames in flight
  // colorFinalLayouts[i] - VK_IMAGE_LAYOUT_GENERAL or VK_IMAGE_LAYOUT_PRESENT_SRC_KHR for swapchain
  void create(const std::vector<VkFormat>& colorFormats,
              const std::vector<bool>& clearColors,
              const std::vector<VkImageLayout>& colorFinalLayouts,
              std::optional<VkFormat> depthFormat,
              bool clearDepth,
              const std::vector<std::vector<VkImageView>>& colorImageViews,
              const std::vector<VkImageView>& depthImageViews,
              VkExtent2D extent);

  void recreateFramebuffers(const std::vector<std::vector<VkImageView>>& colorImageViews,
                             const std::vector<VkImageView>& depthImageViews,
                             VkExtent2D extent);

  VkRenderPass getRenderPass() const noexcept;
  VkFramebuffer getFramebuffer(int index) const noexcept;
  int getFramebufferCount() const noexcept;

  ~RenderPass();
};
}  // namespace RenderGraph
