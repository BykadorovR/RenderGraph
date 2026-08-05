module;

#include <volk.h>

export module Surface;

import Window;
import Instance;

export namespace RenderGraph {
class Surface final {
 private:
  VkSurfaceKHR _surface = VK_NULL_HANDLE;
  const Instance* _instance;

 public:
  Surface(const Window& window, const Instance& instance);
  Surface(const Surface&) = delete;
  Surface& operator=(const Surface&) = delete;
  Surface(Surface&&) = delete;
  Surface& operator=(Surface&&) = delete;

  VkSurfaceKHR getSurface() const noexcept;
  ~Surface();
};
}  // namespace RenderGraph
