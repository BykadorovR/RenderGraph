module;

#include "BS_thread_pool.hpp"

#include <cstddef>
#include <deque>
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
  std::vector<std::unique_ptr<GraphPass>> _passes;
  std::deque<GraphPass*> _passesOrdered;
  std::unordered_map<GraphPass*, QueueType> _passQueueTypes;
  std::unordered_set<GraphPass*> _collectionPasses;
  std::unique_ptr<GraphStorage> _graphStorage;
  // special semaphores
  std::vector<std::shared_ptr<Semaphore>> _semaphoreRenderFinished, _semaphoreImageAvailable;
  std::unique_ptr<Semaphore> _semaphoreInFlight;
  uint64_t _frameNumber = 0;
  int _maxFramesInFlight;

  // resources per pass
  struct Resource {
    std::string name;
    enum class Type { IMAGE, BUFFER } type;
    enum class Operation : uint8_t {
      READ = 1 << 0,
      WRITE = 1 << 1,
    };
    uint8_t operation = 0;
    // Prepared by calculate() from the pass's declared inputs/outputs.
    VkPipelineStageFlags2 stageMask = 0;
    VkAccessFlags2 accessMask = 0;
  };

  class Resources {
   private:
    std::vector<Resource> _resources;

   public:
    void add(Resource resource);
    bool contains(std::string_view name, Resource::Type type, Resource::Operation operation) const;
    const std::vector<Resource>& getResources() const noexcept;
    std::vector<Resource> getResources(Resource::Operation operation) const;
  };

  std::unordered_map<GraphPass*, Resources> _resources;

  // Every submission waits for external GPU work until the pending tickets have been consumed.
  std::vector<GpuTicket> _pendingTickets;
  GraphPass* _swapchainFirstPass = nullptr;

  // A selected physical access and its submission's completion point.
  struct Access {
    VkQueue queue = VK_NULL_HANDLE;
    VkPipelineStageFlags2 stageMask = 0;
    VkAccessFlags2 accessMask = 0;
    uint64_t value = 0;
    Buffer* buffer = nullptr;
    Image* image = nullptr;
    bool writes = false;
  };
  // Per physical instance: the last WRITE (if any), then subsequent READs, one per queue. Kept across frames.
  std::unordered_map<VkBuffer, std::vector<Access>> _bufferHistory;
  std::unordered_map<VkImage, std::vector<Access>> _imageHistory;
  // Same-queue dependencies rebuilt from physical history before recording commands in each render().
  std::unordered_map<GraphPass*, Barrier> _physicalBarriers;

  // If Input and Output select the same buffer/image, combine their READ/WRITE and stage/access masks
  // to build its dependencies and update its history once per pass.
  // calculate() reserves the vector; render() reuses it across passes and frames.
  std::vector<Access> _resolvedAccesses;

  // Cross-queue synchronization: one timeline per real queue assigns completion points to its submissions.
  struct QueueTimeline {
    std::unique_ptr<Semaphore> semaphore;
    uint64_t value = 0;
  };
  std::unordered_map<VkQueue, QueueTimeline> _queueTimelines;

  // Passes in [firstPass, endPass) are sent to queue in one vkQueueSubmit2 call.
  // calculate() groups the passes and prepares commands for each frame slot.
  // render() sets the semaphore waits/signals and the timeline value that marks this submission's completion.
  struct Submission {
    VkQueue queue;
    std::size_t firstPass = 0, endPass = 0;
    std::vector<std::vector<VkCommandBufferSubmitInfo>> commands;
    std::vector<VkSemaphoreSubmitInfo> waits, signals;
    uint64_t value = 0;
  };
  std::vector<Submission> _submissions;

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
