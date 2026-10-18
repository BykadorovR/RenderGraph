module;

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <volk.h>

export module Pipeline;

import Buffer;
import DescriptorBuffer;
import Device;

export namespace RenderGraph {
class PipelineCache final {
 private:
  const Device* _device;
  VkPipelineCache _pipelineCache = VK_NULL_HANDLE;

  VkPipelineCache _create(std::span<const uint8_t> data) const;

 public:
  explicit PipelineCache(const Device& device);
  PipelineCache(const PipelineCache&) = delete;
  PipelineCache& operator=(const PipelineCache&) = delete;
  PipelineCache(PipelineCache&&) = delete;
  PipelineCache& operator=(PipelineCache&&) = delete;

  void setData(std::span<const uint8_t> data);
  std::vector<uint8_t> getData() const;
  const VkPipelineCache getPipelineCache() const noexcept;

  ~PipelineCache();
};

class PipelineGraphic final {
 private:
  VkPipelineDynamicStateCreateInfo _dynamicState;
  std::vector<VkDynamicState> _dynamicStates;
  VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
  VkPipelineViewportStateCreateInfo _viewportState;
  VkPipelineRasterizationStateCreateInfo _rasterizer;
  VkPipelineMultisampleStateCreateInfo _multisampling;
  VkPipelineColorBlendAttachmentState _blendAttachmentState;
  VkPipelineColorBlendStateCreateInfo _colorBlending;
  VkPipelineDepthStencilStateCreateInfo _depthStencil;
  std::optional<VkPipelineTessellationStateCreateInfo> _tessellationState;
  std::vector<VkFormat> _colorAttachments;
  std::optional<VkFormat> _depthAttachment;

 public:
  PipelineGraphic();
  PipelineGraphic(const PipelineGraphic&) = delete;
  PipelineGraphic& operator=(const PipelineGraphic&) = delete;
  PipelineGraphic(PipelineGraphic&&) = delete;
  PipelineGraphic& operator=(PipelineGraphic&&) = delete;

  void setCullMode(VkCullModeFlags cullMode) noexcept;
  void setPolygonMode(VkPolygonMode polygonMode) noexcept;
  void setAlphaBlending(bool alphaBlending) noexcept;
  void setTopology(VkPrimitiveTopology topology) noexcept;
  void setDepthBias(bool depthBias) noexcept;
  void setDepthTest(bool depthTest) noexcept;
  void setDepthWrite(bool depthWrite) noexcept;
  void setDepthCompareOp(VkCompareOp depthCompareOp) noexcept;
  void setColorBlendOp(VkBlendOp colorBlendOp) noexcept;
  void setTesselation(int patchControlPoints) noexcept;
  void setColorAttachments(const std::vector<VkFormat>& colorAttachments);
  void setDepthAttachment(std::optional<VkFormat> depthAttachment) noexcept;

  const VkPipelineDynamicStateCreateInfo& getDynamicState() const noexcept;
  const VkPipelineInputAssemblyStateCreateInfo& getInputAssembly() const noexcept;
  const VkPipelineViewportStateCreateInfo& getViewportState() const noexcept;
  const VkPipelineRasterizationStateCreateInfo& getRasterizer() const noexcept;
  const VkPipelineMultisampleStateCreateInfo& getMultisampling() const noexcept;
  const VkPipelineColorBlendAttachmentState& getBlendAttachmentState() const noexcept;
  const VkPipelineColorBlendStateCreateInfo& getColorBlending() const noexcept;
  const VkPipelineDepthStencilStateCreateInfo& getDepthStencil() const noexcept;
  const std::optional<VkPipelineTessellationStateCreateInfo>& getTessellationState() const noexcept;
  const std::vector<VkFormat>& getColorAttachments() const noexcept;
  const std::optional<VkFormat>& getDepthAttachment() const noexcept;
};

class Pipeline final {
 protected:
  const Device* _device;
  const PipelineCache* _pipelineCache = nullptr;
  std::vector<DescriptorSetLayout*> _descriptorSetLayout;
  std::unordered_map<std::string, VkPushConstantRange> _pushConstants;
  VkPipeline _pipeline{};
  VkPipelineLayout _pipelineLayout{};

 public:
  Pipeline(const Device& device) noexcept;
  Pipeline(const Device& device, const PipelineCache& pipelineCache) noexcept;
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline(Pipeline&&) = delete;
  Pipeline& operator=(Pipeline&&) = delete;

  void createCompute(const VkPipelineShaderStageCreateInfo& shaderStage,
                     std::vector<DescriptorSetLayout*>& descriptorSetLayout,
                     const std::unordered_map<std::string, VkPushConstantRange>& pushConstants);
  void createGraphic(const PipelineGraphic& pipelineGraphic,
                     const std::vector<VkPipelineShaderStageCreateInfo>& shaderStages,
                     std::vector<DescriptorSetLayout*>& descriptorSetLayout,
                     const std::unordered_map<std::string, VkPushConstantRange>& pushConstants,
                     const VkPipelineVertexInputStateCreateInfo& vertexInputInfo);

  const std::vector<DescriptorSetLayout*>& getDescriptorSetLayout() const noexcept;
  const std::unordered_map<std::string, VkPushConstantRange>& getPushConstants() const noexcept;
  const VkPipeline& getPipeline() const noexcept;
  const VkPipelineLayout& getPipelineLayout() const noexcept;
  ~Pipeline();
};
}  // namespace RenderGraph
