module;

#if defined(__ANDROID__)
#include <android/native_window.h>
#else
#include <GLFW/glfw3.h>
#endif
#include <stdexcept>
#include <volk.h>

module Surface;

using namespace RenderGraph;

Surface::Surface(const Window& window, const Instance& instance) : _instance(&instance) {
#if defined(__ANDROID__)
  if (!window.getWindow()) throw std::runtime_error("Can't create an Android surface from a null native window");

  const VkAndroidSurfaceCreateInfoKHR createInfo{
      .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
      .window = window.getWindow(),
  };
  if (vkCreateAndroidSurfaceKHR(instance.getInstance(), &createInfo, nullptr, &_surface) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Android window surface");
  }
#else
  if (glfwCreateWindowSurface(instance.getInstance(), window.getWindow(), nullptr, &_surface) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create GLFW window surface");
  }
#endif
}

VkSurfaceKHR Surface::getSurface() const noexcept { return _surface; }

Surface::~Surface() {
  if (_surface) vkDestroySurfaceKHR(_instance->getInstance().instance, _surface, nullptr);
}
