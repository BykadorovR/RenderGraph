module;

#include "BS_thread_pool.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <volk.h>

export module Graph;

import Pipeline;
import Texture;
import Swapchain;
import Sync;
import Command;
import CommandPool;
import Buffer;
import Device;
import Instrumentation;
import glm;

// Keep GoogleTest-generated classes in the global module
extern "C++" {
class ScenarioTest_GraphSeparateQueues_Test;
class ScenarioTest_BufferQueueChangesUseSemaphores_Test;
class ScenarioTest_BufferQueueChangesSurviveFrameSlotReuse_Test;
class ScenarioTest_ComputePassIndirectBufferInput_Test;
class ScenarioTest_GraphicsPassBufferInputs_Test;
class ScenarioTest_GraphOneQueue_Test;
class ScenarioTest_TraversalKeepsTransitiveProducerBeforeConsumer_Test;
class ScenarioTest_TraversalDiamondGraph_Test;
}

export namespace RenderGraph {
enum class ImageResizePolicy { FIXED, SWAPCHAIN };

class GraphStorage final {
 private:
  struct ImageResource {
    std::unique_ptr<ImageViewHolder> imageViewHolder;
    ImageResizePolicy resizePolicy;
  };
  std::unordered_map<std::string, ImageResource> _images;
  std::unordered_map<std::string, std::unique_ptr<BufferHandler>> _buffers;

 public:
  GraphStorage() = default;
  GraphStorage(const GraphStorage&) = delete;
  GraphStorage& operator=(const GraphStorage&) = delete;
  GraphStorage(GraphStorage&&) = delete;
  GraphStorage& operator=(GraphStorage&&) = delete;

  // Non-swapchain images retain their size and contents by default when the swapchain is reset.
  void add(std::string_view name,
           std::unique_ptr<ImageViewHolder> imageViewHolder,
           ImageResizePolicy resizePolicy = ImageResizePolicy::FIXED);
  void add(std::string_view name, std::unique_ptr<BufferHandler> bufferHandler);
  void reset(std::vector<std::shared_ptr<ImageView>> oldSwapchain,
             std::vector<std::shared_ptr<ImageView>> newSwapchain);
  std::string find(const std::vector<std::shared_ptr<ImageView>>& imageViews);
  bool containsImageViewHolder(std::string_view name) const;
  const ImageViewHolder& getImageViewHolder(std::string_view name) const;
  bool containsBufferHandler(std::string_view name) const;
  const BufferHandler& getBufferHandler(std::string_view name) const;
};

class GraphElement {
 public:
  virtual void draw(int currentFrame, const CommandBuffer& commandBuffer) = 0;
  virtual void update(int currentFrame, const CommandBuffer& commandBuffer) = 0;
  virtual void reset(const std::vector<std::shared_ptr<RenderGraph::ImageView>>& swapchain) = 0;
  virtual ~GraphElement() = default;
};

enum class GraphPassType { GRAPHIC, COMPUTE };

enum class RenderTargetLoadAction { CLEAR, KEEP, DONT_CARE };

enum class RenderTargetStoreAction { STORE, DONT_CARE };

class GraphPass {
 protected:
  std::string _name;
  GraphPassType _graphPassType;
  const GraphStorage* _graphStorage;
  std::unique_ptr<CommandPool> _commandPool;
  std::vector<std::unique_ptr<CommandBuffer>> _commandBuffers;
  std::vector<std::shared_ptr<GraphElement>> _graphElements;

 public:
  GraphPass(std::string_view name, GraphPassType graphPassType, const GraphStorage& graphStorage);
  GraphPass(const GraphPass&) = delete;
  GraphPass& operator=(const GraphPass&) = delete;
  GraphPass(GraphPass&&) = delete;
  GraphPass& operator=(GraphPass&&) = delete;

  void registerGraphElement(std::shared_ptr<GraphElement> graphElement);
  // NVRO
  GraphPassType getGraphPassType() const noexcept;
  QueueType getQueueType() const noexcept;
  CommandBuffer& getCommandBuffer(int frame) const;
  std::vector<CommandBuffer*> getCommandBuffers() const;
  std::string getName() const;
  virtual void execute(int currentFrame, const CommandBuffer& commandBuffer) = 0;
  void reset(const std::vector<std::shared_ptr<RenderGraph::ImageView>>& swapchain);
  virtual ~GraphPass() = default;
};

class GraphPassGraphic final : public GraphPass {
 private:
  struct RenderTargetSettings {
    RenderTargetLoadAction loadAction;
    RenderTargetStoreAction storeAction;
    VkClearValue clearValue;
  };

  std::vector<std::string> _colorTargets, _textureInputs, _vertexBufferInputs, _indexBufferInputs;
  std::vector<std::string> _storageBufferInputs, _indirectBufferInputs;
  std::optional<std::string> _depthTarget;
  std::unordered_map<std::string, RenderTargetSettings> _targetSettings;

  std::unique_ptr<PipelineGraphic> _pipelineGraphic;
  const Device* _device;

 public:
  GraphPassGraphic(std::string_view name,
                   int maxFramesInFlight,
                   const GraphStorage& graphStorage,
                   const Device& device);
  GraphPassGraphic(const GraphPassGraphic&) = delete;
  GraphPassGraphic& operator=(const GraphPassGraphic&) = delete;
  GraphPassGraphic(GraphPassGraphic&&) = delete;
  GraphPassGraphic& operator=(GraphPassGraphic&&) = delete;

  // handle attachments
  void addColorTarget(std::string_view name,
                      RenderTargetLoadAction loadAction = RenderTargetLoadAction::CLEAR,
                      RenderTargetStoreAction storeAction = RenderTargetStoreAction::STORE,
                      VkClearValue clearValue = {.color = {{0.0f, 0.0f, 0.0f, 1.0f}}});
  void setDepthTarget(std::string_view name,
                      RenderTargetLoadAction loadAction = RenderTargetLoadAction::CLEAR,
                      RenderTargetStoreAction storeAction = RenderTargetStoreAction::STORE,
                      VkClearValue clearValue = {.depthStencil = {1.0f, 0}});
  // handle input to shaders
  void addTextureInput(std::string_view name);
  void addVertexBufferInput(std::string_view name);
  void addIndexBufferInput(std::string_view name);
  void addStorageBufferInput(std::string_view name);
  // handle input to indirect draw commands
  void addIndirectBufferInput(std::string_view name);

  const std::vector<std::string>& getColorTargets() const noexcept;
  std::optional<std::string> getDepthTarget() const;
  RenderTargetLoadAction getTargetLoadAction(std::string_view name) const;
  const std::vector<std::string>& getTextureInputs() const noexcept;
  const std::vector<std::string>& getVertexBufferInputs() const noexcept;
  const std::vector<std::string>& getIndexBufferInputs() const noexcept;
  const std::vector<std::string>& getStorageBufferInputs() const noexcept;
  const std::vector<std::string>& getIndirectBufferInputs() const noexcept;
  PipelineGraphic& getPipelineGraphic(const GraphStorage& graphStorage) const;
  void execute(int currentFrame, const CommandBuffer& commandBuffer) override;
};

class GraphPassCompute final : public GraphPass {
 private:
  std::vector<std::string> _storageBufferInputs, _storageBufferOutputs, _indirectBufferInputs;
  std::vector<std::string> _storageTextureInputs, _storageTextureOutputs;
  const Device* _device;

 public:
  GraphPassCompute(std::string_view name,
                   int maxFramesInFlight,
                   QueueType queueType,
                   const GraphStorage& graphStorage,
                   const Device& device);
  GraphPassCompute(const GraphPassCompute&) = delete;
  GraphPassCompute& operator=(const GraphPassCompute&) = delete;
  GraphPassCompute(GraphPassCompute&&) = delete;
  GraphPassCompute& operator=(GraphPassCompute&&) = delete;

  // handle input to shaders
  void addStorageBufferInput(std::string_view name);
  void addStorageTextureInput(std::string_view name);
  // handle input to indirect dispatch commands
  void addIndirectBufferInput(std::string_view name);
  // handle output from shaders
  void addStorageBufferOutput(std::string_view name);
  void addStorageTextureOutput(std::string_view name);

  const std::vector<std::string>& getStorageBufferInputs() const noexcept;
  const std::vector<std::string>& getStorageBufferOutputs() const noexcept;
  const std::vector<std::string>& getStorageTextureInputs() const noexcept;
  const std::vector<std::string>& getStorageTextureOutputs() const noexcept;
  const std::vector<std::string>& getIndirectBufferInputs() const noexcept;
  void execute(int currentFrame, const CommandBuffer& commandBuffer) override;
};

class Graph final {
 private:
  friend class ::ScenarioTest_GraphSeparateQueues_Test;
  friend class ::ScenarioTest_BufferQueueChangesUseSemaphores_Test;
  friend class ::ScenarioTest_BufferQueueChangesSurviveFrameSlotReuse_Test;
  friend class ::ScenarioTest_ComputePassIndirectBufferInput_Test;
  friend class ::ScenarioTest_GraphicsPassBufferInputs_Test;
  friend class ::ScenarioTest_GraphOneQueue_Test;
  friend class ::ScenarioTest_TraversalKeepsTransitiveProducerBeforeConsumer_Test;
  friend class ::ScenarioTest_TraversalDiamondGraph_Test;

  Swapchain* _swapchain = nullptr;
  const Device* _device;
  Tracer* _tracer;
  std::unique_ptr<BS::thread_pool> _threadPool;
  // Passes execute in declaration order; grouping submissions does not reorder them.
  std::vector<std::unique_ptr<GraphPass>> _passes;
  // First pass per VkQueue, used for ticket waits and GPU profiling: G0 G1 C0 G2 -> {G0, C0}.
  std::unordered_set<GraphPass*> _queueFirstPasses;
  std::unique_ptr<GraphStorage> _graphStorage;
  // Acquire: one binary semaphore per frame slot. Present-ready: one per swapchain image.
  std::vector<std::shared_ptr<Semaphore>> _semaphoreRenderFinished, _semaphoreImageAvailable;
  // Separate timeline for CPU frame-slot reuse; its signal must cover all graph queues.
  std::unique_ptr<Semaphore> _semaphoreInFlight;
  uint64_t _frameNumber = 0;
  int _maxFramesInFlight;

  // calculate() stores resource names/types/masks; render() selects the physical buffer/image.
  struct Resource {
    std::string name;
    enum class Type { IMAGE, BUFFER } type;
    VkPipelineStageFlags2 stageMask = 0;
    VkAccessFlags2 accessMask = 0;
  };

  class Resources {
   private:
    std::vector<Resource> _resources;

   public:
    void add(Resource resource);
    const std::vector<Resource>& getResources() const noexcept;
  };

  // One description per name in each pass, shared by all frame slots.
  std::unordered_map<GraphPass*, Resources> _resources;

  // Upload/mipmap readiness, plus the previous frame when calculate() rebuilds history; consumed once.
  std::vector<GpuTicket> _pendingTickets;

  // A selected physical access and its submission's completion point.
  struct Access {
    VkQueue queue = VK_NULL_HANDLE;
    VkPipelineStageFlags2 stageMask = 0;
    VkAccessFlags2 accessMask = 0;
    // Queue timeline completion value; zero when only one queue is used and no queue timeline is needed.
    uint64_t value = 0;
    Buffer* buffer = nullptr;
    Image* image = nullptr;
  };
  // Per physical handle: optional last writer, then readers (one per queue).
  // READ keeps the writer and merges reader masks/max value; WRITE replaces the history.
  // Shared across names and frames, including repeated slots like [A,A,B]; cleared by calculate().
  std::unordered_map<VkBuffer, std::vector<Access>> _bufferHistory;
  std::unordered_map<VkImage, std::vector<Access>> _imageHistory;
  // Same-queue dependencies rebuilt from physical history before recording commands in each render().
  std::unordered_map<GraphPass*, Barrier> _physicalBarriers;

  // One semaphore per real queue; its counter increases per submission, not per frame.
  struct QueueTimeline {
    std::unique_ptr<Semaphore> semaphore;
    uint64_t value = 0;
  };
  std::unordered_map<VkQueue, QueueTimeline> _queueTimelines;

  // calculate() groups adjacent passes on one VkQueue; render() submits their current-slot command buffers.
  // [firstPass, endPass) excludes endPass: [0,3) contains passes 0,1,2.
  // waits/signals describe existing semaphores; adding an entry does not create a semaphore or submission.
  struct Submission {
    VkQueue queue;
    std::size_t firstPass = 0, endPass = 0;
    std::vector<VkSemaphoreSubmitInfo> waits, signals;
    uint64_t value = 0;
  };
  std::vector<Submission> _submissions;

  bool _hasReadAccess(VkAccessFlags2 mask) const noexcept;
  bool _hasWriteAccess(VkAccessFlags2 mask) const noexcept;
  void _setResourceNames();
  void _preparePhysicalAccesses();
  void _addWait(std::vector<VkSemaphoreSubmitInfo>& waits,
                VkSemaphore semaphore,
                uint64_t value,
                VkPipelineStageFlags2 stages);

 public:
  Graph(int threadsNumber, int maxFramesInFlight, const Device& device, Tracer& tracer);
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  Graph(Graph&&) = delete;
  Graph& operator=(Graph&&) = delete;

  void setSwapchain(Swapchain& swapchain);
  void initialize();
  GraphPassGraphic& createPassGraphic(std::string_view name);
  // Only GENERAL and COMPUTE queues are supported; other values throw std::invalid_argument.
  GraphPassCompute& createPassCompute(std::string_view name, QueueType queueType);
  GraphPassGraphic* getPassGraphic(std::string_view name) const;
  GraphPassCompute* getPassCompute(std::string_view name) const;
  GraphStorage& getGraphStorage() const noexcept;
  int getFrameInFlight() const noexcept;
  uint64_t getFrameNumber() const noexcept;

  // Keep ticket producers and Buffer/Image objects alive until their last GPU use completes; tickets are copied here.
  void calculate(std::span<const GpuTicket> tickets = {});
  // true -> need to call reset; an empty graph returns false without submitting work.
  bool render();
  void reset(glm::ivec2 resolution);

  void print() const;
};
}  // namespace RenderGraph
