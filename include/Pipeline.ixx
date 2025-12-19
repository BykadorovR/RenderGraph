export module Pipeline;
import Buffer;
import DescriptorBuffer;
import Device;
import <volk.h>;
import <vector>;
import <map>;
import <string>;
import <optional>;
import <ranges>;

export namespace RenderGraph {
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
  PipelineGraphic() noexcept;
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
  void setDepthCompateOp(VkCompareOp depthCompareOp) noexcept;
  void setColorBlendOp(VkBlendOp colorBlendOp) noexcept;
  void setTesselation(int patchControlPoints) noexcept;
  void setColorAttachments(const std::vector<VkFormat>& colorAttachments) noexcept;
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
  std::vector<std::pair<std::string, DescriptorSetLayout*>> _descriptorSetLayout;
  std::map<std::string, VkPushConstantRange> _pushConstants;
  VkPipeline _pipeline;
  VkPipelineLayout _pipelineLayout;

 public:
  Pipeline(const Device& device) noexcept;
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline(Pipeline&&) = delete;
  Pipeline& operator=(Pipeline&&) = delete;

  void createCompute(const VkPipelineShaderStageCreateInfo& shaderStage,
                     std::vector < std::pair<std::string, DescriptorSetLayout*>> & descriptorSetLayout,
                     const std::map<std::string, VkPushConstantRange>& pushConstants);
  void createGraphic(const PipelineGraphic& pipelineGraphic,
                     const std::vector<VkPipelineShaderStageCreateInfo>& shaderStages,
                     std::vector<std::pair<std::string, DescriptorSetLayout*>>& descriptorSetLayout,
                     const std::map<std::string, VkPushConstantRange>& pushConstants,
                     const VkPipelineVertexInputStateCreateInfo& vertexInputInfo);

  const std::vector<std::pair<std::string, DescriptorSetLayout*>>& getDescriptorSetLayout() const noexcept;
  const std::map<std::string, VkPushConstantRange>& getPushConstants() const noexcept;
  const VkPipeline& getPipeline() const noexcept;
  const VkPipelineLayout& getPipelineLayout() const noexcept;
  ~Pipeline();
};
}  // namespace RenderGraph