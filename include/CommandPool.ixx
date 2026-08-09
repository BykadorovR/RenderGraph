module;
#include <volk.h>

export module CommandPool;

import Device;

export namespace RenderGraph {
class CommandPool final {
 private:
  const Device* _device;
  QueueType _type;
  VkCommandPool _commandPool;

 public:
  CommandPool(QueueType type, const Device& device);
  CommandPool(const CommandPool&) = delete;
  CommandPool& operator=(const CommandPool&) = delete;
  CommandPool(CommandPool&&) = delete;
  CommandPool& operator=(CommandPool&&) = delete;

  VkCommandPool getCommandPool() const noexcept;
  QueueType getType() const noexcept;
  ~CommandPool();
};
}  // namespace RenderGraph
