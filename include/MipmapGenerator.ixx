module;

#include <cstdint>
#include <deque>
#include <memory>
#include <span>
#include <volk.h>

export module MipmapGenerator;

import Command;
import CommandPool;
import Device;
import Sync;
import Texture;

export namespace RenderGraph {
class MipmapGenerator final {
 private:
  struct GenerationBatch {
    std::unique_ptr<CommandBuffer> _commandBuffer;
    uint64_t _completionValue = 0;

    GenerationBatch(const CommandPool& commandPool, const Device& device);
  };

  const Device* _device;
  CommandPool _commandPool;
  Semaphore _timelineSemaphore;
  std::unique_ptr<GenerationBatch> _recordingBatch;
  std::deque<std::unique_ptr<GenerationBatch>> _inFlightBatches;
  uint64_t _nextCompletionValue = 1;

  GenerationBatch& _getRecordingBatch();

 public:
  // All generation commands execute on GENERAL.
  explicit MipmapGenerator(const Device& device);
  MipmapGenerator(const MipmapGenerator&) = delete;
  MipmapGenerator& operator=(const MipmapGenerator&) = delete;
  MipmapGenerator(MipmapGenerator&&) = delete;
  MipmapGenerator& operator=(MipmapGenerator&&) = delete;

  // Generate levels 1..N from the initialized base level for every layer, using the requested filter.
  void generate(Image& image, VkFilter filter = VK_FILTER_LINEAR);
  // Wait for input work on the GPU; the returned ticket covers this batch's generation.
  GpuTicket submit(std::span<const GpuTicket> waits = {});
  void reclaim();

  ~MipmapGenerator();
};
}  // namespace RenderGraph
