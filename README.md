# RenderGraph

RenderGraph is a lightweight C++ library for organizing rendering and compute workloads with Vulkan. It helps describe passes and resources while handling execution order and synchronization behind the scenes.

The project is designed for applications that need a compact rendering foundation without adopting a full game engine.

## Features

- Graphics and compute passes
- Fully built around C++ modules
- Automatic pass ordering and resource synchronization
- Multi-queue execution and resource ownership transfers
- Headless compute workloads
- Descriptor buffers and descriptor sets
- Asynchronous CPU-to-GPU resource uploads
- GPU timestamp collection
- Desktop and Android support

## Requirements

- CMake 3.28 or newer
- A C++23-compatible compiler with C++ modules support
- Vulkan SDK and a Vulkan 1.3-capable device
- `glslangValidator` when building the test suite

Third-party libraries are downloaded automatically by CMake.

## Using RenderGraph in a CMake Project

RenderGraph can be added directly with CMake's `FetchContent`:

```cmake
include(FetchContent)

set(RENDER_GRAPH_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  RenderGraph
  GIT_REPOSITORY https://github.com/BykadorovR/RenderGraph.git
  GIT_TAG master
)

FetchContent_MakeAvailable(RenderGraph)

target_link_libraries(yourTarget PRIVATE RenderGraph::RenderGraph)
```

CMake downloads RenderGraph and its third-party dependencies during configuration. The library's C++ modules can then be imported directly from the linked target.

## Headless Example

The following example creates a headless Vulkan device and an empty compute graph:

```cpp
import Device;
import Graph;
import Instance;

int main() {
  RenderGraph::Instance instance("Example", true);

  RenderGraph::Device device(instance);
  device.initialize();

  // Worker threads used to record pass commands into Vulkan command buffers in parallel.
  constexpr int workerThreadCount = 2;
  // Frames that may be processed concurrently.
  constexpr int framesInFlight = 2;

  RenderGraph::Graph graph(workerThreadCount, framesInFlight, device);
  graph.initialize();

  // Adds a compute pass to the graphics queue instead of a separate queue.
  graph.createPassCompute("Compute", false);
  graph.calculate();
  // Call once per frame from a dedicated rendering thread.
  graph.render();
}
```

This minimal example submits only one frame. In a real application, call `render()` for every frame from a dedicated rendering thread.

## Window and Swapchain Example

This example creates a window and runs a small three-pass rendering graph:

```cpp
#define VK_NO_PROTOTYPES
#define GLFW_INCLUDE_NONE

#include <GLFW/glfw3.h>
#include <memory>
#include <utility>
#include <volk.h>

import Allocator;
import Device;
import Graph;
import Instance;
import Surface;
import Swapchain;
import Texture;
import Window;
import glm;

void buildGraph(RenderGraph::Graph& graph, RenderGraph::Swapchain& swapchain) {
  graph.setSwapchain(swapchain);
  graph.initialize();

  // Exposes the currently acquired swapchain image as a graph resource.
  auto swapchainImages = std::make_unique<RenderGraph::ImageViewHolder>(
      swapchain.getImageViews(), [&swapchain]() { return swapchain.getSwapchainIndex(); });
  graph.getGraphStorage().add("Swapchain", std::move(swapchainImages));

  auto& clearPass = graph.createPassGraphic("Clear");
  clearPass.addColorTarget("Swapchain");
  clearPass.clearTarget("Swapchain");

  auto& scenePass = graph.createPassGraphic("Scene");
  scenePass.addColorTarget("Swapchain");
  // Application-provided GraphElement implementations record the actual draw commands.
  // scenePass.registerGraphElement(sceneElement);

  auto& uiPass = graph.createPassGraphic("UI");
  uiPass.addColorTarget("Swapchain");
  // uiPass.registerGraphElement(uiElement);

  graph.calculate();
}

void runRenderLoop(RenderGraph::Graph& graph, RenderGraph::Window& window) {
  while (!glfwWindowShouldClose(window.getWindow())) {
    glfwPollEvents();
    // render() returns true when swapchain-dependent resources must be recreated.
    if (graph.render()) {
      const glm::ivec2 resolution = window.getResolution();
      if (resolution.x > 0 && resolution.y > 0) {
        graph.reset(resolution);
      }
    }
  }
}

int main() {
  const glm::ivec2 resolution{1280, 720};

  RenderGraph::Instance instance("Example", true);
  RenderGraph::Window window(resolution);
  window.initialize();
  RenderGraph::Surface surface(window, instance);

  RenderGraph::Device device(instance);
  device.setSurface(surface);
  device.initialize();

  RenderGraph::MemoryAllocator allocator(device, instance);
  RenderGraph::Swapchain swapchain(resolution, allocator, device);
  swapchain.initialize();

  // Worker threads used to record pass commands into Vulkan command buffers in parallel.
  constexpr int workerThreadCount = 2;
  // Frames that may be processed concurrently.
  constexpr int framesInFlight = 2;
  RenderGraph::Graph graph(workerThreadCount, framesInFlight, device);

  buildGraph(graph, swapchain);
  runRenderLoop(graph, window);

  // Waits for submitted GPU work before resources leave scope.
  vkDeviceWaitIdle(device.getLogicalDevice());
}
```

[`GraphElement`](include/Graph.ixx#L64-L70) is the application-provided callback interface with `update()`, `draw()`, and `reset()` methods. The two commented calls show where those implementations are attached. Without them, `Scene` and `UI` contain no draw commands, and this example only clears and presents the swapchain image. See the [render graph scenarios](tests/Scenarios.cpp) for complete registrations.

## Used By

RenderGraph is used by [VulkanEngine](https://github.com/BykadorovR/VulkanEngine).

## Project Status

RenderGraph is under active development. Its API may change as new rendering workflows and platforms are added.

## License

RenderGraph is licensed under the MIT License.
