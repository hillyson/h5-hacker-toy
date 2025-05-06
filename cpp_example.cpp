#include <memory>
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <future>
#include <chrono>
#include <type_traits>
#include <coroutine>
#include <filesystem>
#include <concepts>
#include <boost/asio.hpp>
#include <benchmark/benchmark.h>
#include <execution> // Parallel algorithms
#include <memory_resource>
#include <array>
#include <numeric>
#include <tuple>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

template<int N>
struct Factorial {
    static const int value = N * Factorial<N-1>::value;
};

template<>
struct Factorial<0> {
    static const int value = 1;
};

class Resource {
    int* data;
public:
    Resource() : data(new int[100]) { std::cout << "Resource allocated\n"; }
    ~Resource() { delete[] data; std::cout << "Resource freed\n"; }
   
    Resource(Resource&& other) noexcept : data(other.data) {
        other.data = nullptr;
        std::cout << "Resource moved\n";
    }
   
    Resource& operator=(Resource&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            other.data = nullptr;
            std::cout << "Resource move assigned\n";
        }
        return *this;
    }
};

template<typename T>
class ThreadSafeQueue {
    std::queue<T> queue;
    std::mutex mtx;
    std::condition_variable cv;
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push(std::move(value));
        cv.notify_one();
    }
   bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) return false;
        value = std::move(queue.front());
        queue.pop();
        return true;
    }
};

class MyClass {
public:
    MyClass() { std::cout << "MyClass constructed\n"; }
    ~MyClass() { std::cout << "MyClass destroyed\n"; }
    void doSomething() { std::cout << "Doing something\n"; }
};

void optimizedAlgorithm(std::vector<int>& data) {
   for (size_t i = 0; i < data.size(); ++i) {
        data[i] *= 2;
    }
}

task<int> example_coroutine() {
    std::cout << "Coroutine started\n";
    co_await std::suspend_always{};
    std::cout << "Coroutine resumed\n";
    co_return 42;
}

template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::same_as<T>;
};
template<Addable T>
T add(T a, T b) {
    return a + b;
}

void list_files(const std::string& path) {
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        std::cout << entry.path() << std::endl;
    }
}
int main() {
   {
        std::unique_ptr<MyClass> ptr1(new MyClass());
        ptr1->doSomething();
      
std::shared_ptr<MyClass> ptr2 = std::make_shared<MyClass>();
        std::shared_ptr<MyClass> ptr3 = ptr2;
        ptr2->doSomething();
        ptr3->doSomething();
       
  
std::weak_ptr<MyClass> weakPtr = ptr2;
        if (auto sharedPtr = weakPtr.lock()) {
            sharedPtr->doSomething();
        }
    }
   
    std::cout << "Factorial of 5: " << Factorial<5>::value << std::endl;
   
    Resource res1;
    Resource res2 = std::move(res1);
   
    auto lambda = [](int x) { return x * x; };
    std::cout << "Lambda result: " << lambda(5) << std::endl;
   
    std::vector<int> numbers = {3, 1, 4, 1, 5, 9, 2, 6};
    std::sort(numbers.begin(), numbers.end());
    std::for_each(numbers.begin(), numbers.end(), [](int n) {
        std::cout << n << " ";
    });
    std::cout << std::endl;
   
    std::vector<std::thread> threads;
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([i]() {
            std::cout << "Thread " << i << " running\n";
        });
    }
   for (auto& t : threads) {
        t.join();
    }
   
    auto future = std::async(std::launch::async, []() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return "Async task completed";
    });
   std::cout << future.get() << std::endl;
   
    std::vector<int> bigData(1000000, 1);
    optimizedAlgorithm(bigData);
   
    auto coro = example_coroutine();
    coro.resume();
    std::cout << "Coroutine result: " << coro.get_result() << std::endl;
   
    std::cout << "Addable concept: " << add(5, 7) << std::endl;
   
    std::cout << "Current directory files:" << std::endl;
    list_files(".");
   
    boost::asio::io_context io;
    boost::asio::ip::tcp::socket socket(io);
    boost::asio::ip::tcp::resolver resolver(io);
    boost::asio::connect(socket, resolver.resolve("example.com", "80"));
   
    static void BM_StringCreation(benchmark::State& state) {
        for (auto _ : state) {
            std::string empty_string;
        }
    }
    BENCHMARK(BM_StringCreation);
   
    std::vector<int> vec(1000000);
    std::iota(vec.begin(), vec.end(), 0);
    std::for_each(std::execution::par, vec.begin(), vec.end(), [](int& n) {
        n *= 2;
    });
   
    struct alignas(64) CustomObject {
        int data[16];
    };
   std::pmr::monotonic_buffer_resource pool;
    std::pmr::polymorphic_allocator<CustomObject> alloc(&pool);
   
    constexpr auto compileTimeStr = [] {
        constexpr std::string_view sv = "Hello Compile-Time";
        std::array<char, sv.size() + 1> arr{};
        std::copy(sv.begin(), sv.end(), arr.begin());
        return arr;
    }();
   
    #ifdef __CUDACC__
    __global__ void addKernel(int* a, int* b, int* c) {
        int i = threadIdx.x;
        c[i] = a[i] + b[i];
    }
   void gpuExample() {
        const int N = 256;
        int *a, *b, *c;
        cudaMallocManaged(&a, N*sizeof(int));
        cudaMallocManaged(&b, N*sizeof(int));
        cudaMallocManaged(&c, N*sizeof(int));
      
addKernel<<<1, N>>>(a, b, c);
        cudaDeviceSynchronize();
      
cudaFree(a);
        cudaFree(b);
        cudaFree(c);
    }
    #endif
   
    template <typename T>
    class CustomAllocator {
    public:
        using value_type = T;
      
CustomAllocator() = default;
      
template <typename U>
        CustomAllocator(const CustomAllocator<U>&) {}
      
T* allocate(std::size_t n) {
            return static_cast<T*>(::operator new(n * sizeof(T)));
        }
      
void deallocate(T* p, std::size_t) {
            ::operator delete(p);
        }
    };
   
    template <typename... Ts>
    struct TypeList {};
   template <typename List>
    struct Front;
   template <typename Head, typename... Tail>
    struct Front<TypeList<Head, Tail...>> {
        using type = Head;
    };
   
    auto [x, y] = std::make_tuple(1, 2.0);
   
    #if __cpp_reflection >= 202211
    constexpr auto type_name = std::meta::get_type_name<std::vector<int>>();
    std::cout << "Type name: " << type_name << std::endl;
    #endif
   
    auto generator = []() -> generator<int> {
        for (int i = 0; ; ++i) {
            co_yield i;
        }
    };
   
    constexpr auto hash = [] {
        constexpr std::string_view str = "Compile-Time Hash";
        size_t result = 0;
        for (char c : str) {
            result = (result * 131) + c;
        }
        return result;
    }();
   
    auto operator""_s(const char* str, size_t) {
        return std::string(str);
    }
   
    #ifdef __cpp_modules
    import std.core;
    #endif
   
    struct AnyCallable {
        template<typename F>
        AnyCallable(F&& f) : ptr(new Model<F>(std::forward<F>(f))) {}
      
void operator()() const { ptr->call(); }
      
struct Concept {
            virtual ~Concept() = default;
            virtual void call() const = 0;
        };
      
template<typename F>
        struct Model : Concept {
            F f;
            Model(F&& f) : f(std::forward<F>(f)) {}
            void call() const override { f(); }
        };
      
std::unique_ptr<Concept> ptr;
    };
   
    constexpr int factorial(int n) {
        return n <= 1 ? 1 : n * factorial(n - 1);
    }
   
    void structured_concurrency() {
        std::stop_source stop;
        std::jthread worker1([&](std::stop_token st) {
            while (!st.stop_requested()) {
                std::cout << "Worker 1 running\n";
                std::this_thread::sleep_for(1s);
            }
        }, stop.get_token());
      
std::jthread worker2([&](std::stop_token st) {
            while (!st.stop_requested()) {
                std::cout << "Worker 2 running\n";
                std::this_thread::sleep_for(500ms);
            }
        }, stop.get_token());
      
std::this_thread::sleep_for(3s);
        stop.request_stop();
    }
   
    template<typename T>
    constexpr bool is_arithmetic_v = std::is_arithmetic_v<T>;
   
    template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    void process_integer(T value) {
        std::cout << "Processing integer: " << value << std::endl;
    }
   
    std::atomic<int> counter(0);
    std::thread t1([&] { counter.fetch_add(1, std::memory_order_relaxed); });
    std::thread t2([&] { counter.fetch_add(1, std::memory_order_relaxed); });
    t1.join();
    t2.join();
    std::cout << "Counter value: " << counter << std::endl;
   
    struct Base {
        virtual Base* clone() const = 0;
        virtual ~Base() = default;
    };
   struct Derived : Base {
        Derived* clone() const override { return new Derived(*this); }
    };
   
    struct Point { int x, y; };
    Point p{1, 2};
    auto [x, y] = p;
   
    constexpr auto concat = [] {
        constexpr std::string_view s1 = "Hello";
        constexpr std::string_view s2 = " World";
        std::array<char, s1.size() + s2.size() + 1> result{};
        std::copy(s1.begin(), s1.end(), result.begin());
        std::copy(s2.begin(), s2.end(), result.begin() + s1.size());
        return result;
    }();
   
    template<template<typename> class Container, typename T>
    void process_container(Container<T>& c) {
        for (const auto& item : c) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
   
    std::vector<int> nums{1, 2, 3, 4, 5};
    auto even = nums | std::views::filter([](int n) { return n % 2 == 0; });
    for (int n : even) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
   
    generator<int> range(int start, int end) {
        for (int i = start; i < end; ++i) {
            co_yield i;
        }
    }
   
    #if __cpp_reflection >= 202211
    constexpr auto members = std::meta::get_data_members<std::meta::reflect<Point>>();
    for (const auto& member : members) {
        std::cout << std::meta::get_name(member) << std::endl;
    }
    #endif
   
    constexpr auto sorted_array = [] {
        std::array<int, 5> arr{5, 3, 1, 4, 2};
        std::sort(arr.begin(), arr.end());
        return arr;
    }();
   
    template<typename T>
    concept Printable = requires(T t) {
        { std::cout << t } -> std::same_as<std::ostream&>;
    };
   template<Printable T>
    void print(const T& t) {
        std::cout << t << std::endl;
    }
   return 0;
}