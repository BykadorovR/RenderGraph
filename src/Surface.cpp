module Surface;
using namespace RenderGraph;

Surface::Surface(const Window& window, const Instance& instance) : _instance(&instance) {
  if (glfwCreateWindowSurface(instance.getInstance(), window.getWindow(), nullptr, &_surface) != VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface!");
  }
}

VkSurfaceKHR Surface::getSurface() const noexcept { return _surface; }

Surface::~Surface() { vkDestroySurfaceKHR(_instance->getInstance().instance, _surface, nullptr); }