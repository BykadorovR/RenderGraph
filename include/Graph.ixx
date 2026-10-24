module;

#include "BS_thread_pool.hpp"

#include <deque>
#include <functional>
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
import ResourceUploader;
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
  std::unordered_map<std::string, std::vector<std::unique_ptr<Buffer>>> _buffers;

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
  // not const because will do std::move
  void add(std::string_view name, std::vector<std::unique_ptr<Buffer>>& buffers);
  void reset(std::vector<std::shared_ptr<ImageView>> oldSwapchain,
             std::vector<std::shared_ptr<ImageView>> newSwapchain);
  std::string find(const std::vector<std::shared_ptr<ImageView>>& imageViews);
  bool containsImageViewHolder(std::string_view name) const;
  const ImageViewHolder& getImageViewHolder(std::string_view name) const;
  bool containsBuffer(std::string_view name) const;
  // NVRO
  std::vector<Buffer*> getBuffer(std::string_view name) const;
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
  uint64_t _valueSemaphoreInFlight = 1;
  int _maxFramesInFlight;
  int _frameInFlight = 0;

  // resources per pass
  struct Resource {
    std::string name;
    enum class Type { IMAGE, BUFFER } type;
    enum class Operation : uint8_t {
      READ = 1 << 0,
      WRITE = 1 << 1,
    };
    uint8_t operation = 0;
  };

  class Resources {
   private:
    std::vector<Resource> _resources;

   public:
    void add(Resource resource);
    bool contains(std::string_view name, Resource::Type type, Resource::Operation operation) const;
    const std::vector<Resource>& getResources() const noexcept;
    std::vector<Resource> getResources(Resource::Operation operation) const;
    std::vector<std::string> getNames(Resource::Type type) const;
  };

  std::unordered_map<GraphPass*, Resources> _resources;

  // syncrhonization per pass
  class Sync {
   private:
    // same semaphore is wait and signal, so can't be unique
    std::vector<std::pair<std::vector<std::shared_ptr<Semaphore>>, std::function<int()>>> _waitSemaphores,
        _signalSemaphores;
    std::vector<std::pair<std::vector<Barrier>, std::function<int()>>> _barriers;

   public:
    Sync() = default;
    void addSignalSemaphore(std::vector<std::shared_ptr<Semaphore>>& signalSemaphore, std::function<int()> index);
    void addWaitSemaphore(std::vector<std::shared_ptr<Semaphore>>& waitSemaphore, std::function<int()> index);
    void addBarrier(std::vector<Barrier>& barriers, std::function<int()> index);

    std::vector<Semaphore*> getWaitSemaphores() const;
    std::vector<Semaphore*> getSignalSemaphores() const;
    std::vector<const Barrier*> getBarriers() const;
  };

  std::unordered_map<GraphPass*, Sync> _sync;
  // Each queue consumes its upload waits once, at its first pass.
  std::unordered_map<GraphPass*, std::vector<UploadTicket>> _uploadTickets;
  std::unordered_map<GraphPass*, std::vector<std::pair<Image*, Barrier>>> _initialImageBarriers;
  GraphPass* _swapchainFirstPass = nullptr;
  VkPipelineStageFlags2 _swapchainFirstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  VkAccessFlags2 _swapchainFirstAccessMask = 0;

  void _setResourceNames();

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

  // Keep ResourceUploader alive until the GPU finishes the frame waiting on its tickets (first frame-slot reuse).
  // Keep uploaded Buffer/Image objects alive until their last GPU use completes; tickets are copied here.
  void calculate(std::span<const UploadTicket> uploadTickets = {});
  // true -> need to call reset; an empty graph returns false without submitting work.
  bool render();
  void reset(glm::ivec2 resolution);

  void print() const;
};
}  // namespace RenderGraph
