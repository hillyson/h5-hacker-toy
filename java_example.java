class Node<T> {
    T data;
    Node<T> next;
    
    public Node(T data) {
        this.data = data;
        this.next = null;
    }
}

@SpringBootApplication
@RestController
public class DemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }
    @GetMapping("/hello")
    public String hello(@RequestParam(value = "name", defaultValue = "World") String name) {
        return String.format("Hello %s!", name);
    }
}

class ThreadSafeCounter {
    private int count = 0;
    private final Object lock = new Object();
    public void increment() {
        synchronized(lock) {
            count++;
        }
    }
    public int getCount() {
        synchronized(lock) {
            return count;
        }
    }
}
class LinkedList<T> {
    private Node<T> head;
    private Node<T> tail;
    private int size;
    
    public LinkedList() {
        this.head = null;
        this.tail = null;
        this.size = 0;
    }
    
    public void append(T data) {
        Node<T> newNode = new Node<>(data);
        if (head == null) {
            head = newNode;
            tail = newNode;
        } else {
            tail.next = newNode;
            tail = newNode;
        }
        size++;
    }
    
    public void prepend(T data) {
        Node<T> newNode = new Node<>(data);
        if (head == null) {
            head = newNode;
            tail = newNode;
        } else {
            newNode.next = head;
            head = newNode;
        }
        size++;
    }
    
    public void delete(T data) {
        if (head == null) return;
        
        if (head.data.equals(data)) {
            head = head.next;
            size--;
            return;
        }
        
        Node<T> current = head;
        while (current.next != null) {
            if (current.next.data.equals(data)) {
                current.next = current.next.next;
                size--;
                return;
            }
            current = current.next;
        }
    }
    
    public Node<T> search(T data) {
        Node<T> current = head;
        while (current != null) {
            if (current.data.equals(data)) {
                return current;
            }
            current = current.next;
        }
        return null;
    }
    
    public int size() {
        return size;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        Node<T> current = head;
        while (current != null) {
            sb.append(current.data).append(" -> ");
            current = current.next;
        }
        sb.append("null");
        return sb.toString();
    }
}

class Sorting {
    public static void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
    
    public static int binarySearch(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target) {
                return mid;
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }
}

import java.io.*;
import java.nio.file.*;
class FileUtils {
    public static String readFile(String filename) throws IOException {
        return new String(Files.readAllBytes(Paths.get(filename)));
    }
    
    public static void writeFile(String filename, String content) throws IOException {
        Files.write(Paths.get(filename), content.getBytes());
    }
}

import java.net.*;
import java.io.*;
class HttpClient {
    public static String get(String url) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) new URL(url).openConnection();
        connection.setRequestMethod("GET");
        
        BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()));
        String inputLine;
        StringBuilder content = new StringBuilder();
        
        while ((inputLine = in.readLine()) != null) {
            content.append(inputLine);
        }
        in.close();
        connection.disconnect();
        
        return content.toString();
    }
}

import java.sql.*;
class Database {
    private Connection connection;
    
    // Connection Pool Example
    private static DataSource createConnectionPool() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:mysql://localhost:3306/mydb");
        config.setUsername("user");
        config.setPassword("password");
        config.setMaximumPoolSize(10);
        return new HikariDataSource(config);
    }
    
    // JPA Entity Example
    @Entity
    @Table(name = "users")
    public class User {
        @Id
        @GeneratedValue(strategy = GenerationType.IDENTITY)
        private Long id;
        
        @Column(nullable = false)
        private String name;
        
        @Column(unique = true)
        private String email;
        
        // Getters and setters
    }
    
    // Spring Data JPA Repository
    public interface UserRepository extends JpaRepository<User, Long> {
        List<User> findByName(String name);
    }
    
    // Transaction Management Example
    @Transactional
    public void transferMoney(Account from, Account to, BigDecimal amount) {
        from.withdraw(amount);
        to.deposit(amount);
    }
    
    // QueryDSL Example
    public List<User> findActiveUsers() {
        QUser user = QUser.user;
        return new JPAQueryFactory(entityManager)
            .selectFrom(user)
            .where(user.active.eq(true))
            .fetch();
    }
    
    // Redis Cache Example
    @Cacheable(value = "users", key = "#userId")
    public User getUser(Long userId) {
        return userRepository.findById(userId).orElse(null);
    }
    
    // Elasticsearch Example
    @Document(indexName = "products")
    public class Product {
        @Id
        private String id;
        
        @Field(type = FieldType.Text)
        private String name;
        
        // Other fields and methods
    }
    
    // MongoDB Example
    @Document(collection = "orders")
    public class Order {
        @Id
        private String orderId;
        
        private List<OrderItem> items;
        
        // Other fields and methods
    }
    
    // GraphQL Example
    @GraphQLQuery(name = "users")
    public List<User> getUsers() {
        return userRepository.findAll();
    }
    
    // Kafka Producer Example
    public void sendMessage(String topic, String message) {
        kafkaTemplate.send(topic, message);
    }
    
    // RabbitMQ Example
    @RabbitListener(queues = "myQueue")
    public void processMessage(String message) {
        // Process message
    }
    
    // WebSocket Example
    @MessageMapping("/chat")
    @SendTo("/topic/messages")
    public ChatMessage sendMessage(ChatMessage message) {
        return message;
    }
    
    // OAuth2 Security Example
    @EnableWebSecurity
    @EnableGlobalMethodSecurity(prePostEnabled = true)
    public class SecurityConfig extends WebSecurityConfigurerAdapter {
        @Override
        protected void configure(HttpSecurity http) throws Exception {
            http.authorizeRequests()
                .antMatchers("/api/public/**").permitAll()
                .anyRequest().authenticated()
                .and()
                .oauth2Login();
        }
    }
    
    public Database(String url) throws SQLException {
        this.connection = DriverManager.getConnection(url);
    }
    
    public void createTable(String sql) throws SQLException {
        try (Statement stmt = connection.createStatement()) {
            stmt.execute(sql);
        }
    }
    
    public void insert(String table, String[] columns, Object[] values) throws SQLException {
        String placeholders = String.join(", ", java.util.Collections.nCopies(columns.length, "?"));
        String sql = String.format("INSERT INTO %s (%s) VALUES (%s)", 
            table, String.join(", ", columns), placeholders);
            
        try (PreparedStatement pstmt = connection.prepareStatement(sql)) {
            for (int i = 0; i < values.length; i++) {
                pstmt.setObject(i + 1, values[i]);
            }
            pstmt.executeUpdate();
        }
    }
    
    public void close() throws SQLException {
        if (connection != null) {
            connection.close();
        }
    }
}
// Multithreading
class Worker implements Runnable {
    private String name;
    
    public Worker(String name) {
        this.name = name;
    }
    
    @Override
    public void run() {
        System.out.println(name + " is working");
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println(name + " finished");
    }
}
// Thread Pool Example
class ThreadPool {
    private final BlockingQueue<Runnable> taskQueue;
    private final List<WorkerThread> threads;
    private volatile boolean isStopped;
    
    public ThreadPool(int numThreads) {
        this.taskQueue = new LinkedBlockingQueue<>();
        this.threads = new ArrayList<>();
        this.isStopped = false;
        
        for (int i = 0; i < numThreads; i++) {
            threads.add(new WorkerThread(taskQueue));
        }
        
        for (WorkerThread thread : threads) {
            thread.start();
        }
    }
    
    public void execute(Runnable task) {
        if (isStopped) {
            throw new IllegalStateException("ThreadPool is stopped");
        }
        taskQueue.offer(task);
    }
    
    public void stop() {
        isStopped = true;
        for (WorkerThread thread : threads) {
            thread.doStop();
        }
    }
}
class WorkerThread extends Thread {
    private final BlockingQueue<Runnable> taskQueue;
    private volatile boolean isStopped;
    
    public WorkerThread(BlockingQueue<Runnable> taskQueue) {
        this.taskQueue = taskQueue;
        this.isStopped = false;
    }
    
    public void run() {
        while (!isStopped) {
            try {
                Runnable task = taskQueue.take();
                task.run();
            } catch (InterruptedException e) {
                // Handle interruption
            }
        }
    }
    
    public void doStop() {
        isStopped = true;
        this.interrupt();
    }
}
// Observer Pattern
interface Subject {
    void registerObserver(Observer o);
    void removeObserver(Observer o);
    void notifyObservers();
}
interface Observer {
    void update(float temp, float humidity, float pressure);
}
class WeatherData implements Subject {
    private List<Observer> observers;
    private float temperature;
    private float humidity;
    private float pressure;
    
    public WeatherData() {
        observers = new ArrayList<>();
    }
    
    public void registerObserver(Observer o) {
        observers.add(o);
    }
    
    public void removeObserver(Observer o) {
        observers.remove(o);
    }
    
    public void notifyObservers() {
        for (Observer observer : observers) {
            observer.update(temperature, humidity, pressure);
        }
    }
    
    public void measurementsChanged() {
        notifyObservers();
    }
    
    public void setMeasurements(float temperature, float humidity, float pressure) {
        this.temperature = temperature;
        this.humidity = humidity;
        this.pressure = pressure;
        measurementsChanged();
    }
}
class CurrentConditionsDisplay implements Observer {
    private float temperature;
    private float humidity;
    private Subject weatherData;
    
    public CurrentConditionsDisplay(Subject weatherData) {
        this.weatherData = weatherData;
        weatherData.registerObserver(this);
    }
    
    public void update(float temperature, float humidity, float pressure) {
        this.temperature = temperature;
        this.humidity = humidity;
        display();
    }
    
    public void display() {
        System.out.println("Current conditions: " + temperature 
            + "F degrees and " + humidity + "% humidity");
    }
}
// Red-Black Tree
class RedBlackTree<K extends Comparable<K>, V> {
    private static final boolean RED = true;
    private static final boolean BLACK = false;
    
    private class Node {
        K key;
        V value;
        Node left, right;
        boolean color;
        
        Node(K key, V value, boolean color) {
            this.key = key;
            this.value = value;
            this.color = color;
        }
    }
    
    private Node root;
    
    private boolean isRed(Node x) {
        if (x == null) return false;
        return x.color == RED;
    }
    
    private Node rotateLeft(Node h) {
        Node x = h.right;
        h.right = x.left;
        x.left = h;
        x.color = h.color;
        h.color = RED;
        return x;
    }
    
    private Node rotateRight(Node h) {
        Node x = h.left;
        h.left = x.right;
        x.right = h;
        x.color = h.color;
        h.color = RED;
        return x;
    }
    
    private void flipColors(Node h) {
        h.color = !h.color;
        h.left.color = !h.left.color;
        h.right.color = !h.right.color;
    }
    
    public void put(K key, V value) {
        root = put(root, key, value);
        root.color = BLACK;
    }
    
    private Node put(Node h, K key, V value) {
        if (h == null) return new Node(key, value, RED);
        
        int cmp = key.compareTo(h.key);
        if (cmp < 0) h.left = put(h.left, key, value);
        else if (cmp > 0) h.right = put(h.right, key, value);
        else h.value = value;
        
        if (isRed(h.right) && !isRed(h.left)) h = rotateLeft(h);
        if (isRed(h.left) && isRed(h.left.left)) h = rotateRight(h);
        if (isRed(h.left) && isRed(h.right)) flipColors(h);
        
        return h;
    }
    
    public V get(K key) {
        Node x = root;
        while (x != null) {
            int cmp = key.compareTo(x.key);
            if (cmp < 0) x = x.left;
            else if (cmp > 0) x = x.right;
            else return x.value;
        }
        return null;
    }
}
// Testing
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;
class LinkedListTest {
    private LinkedList<String> list;
    
    @BeforeEach
    void setUp() {
        list = new LinkedList<>();
    }
    
    @Test
    void testAppend() {
        list.append("A");
        assertEquals(1, list.size());
        assertEquals("A -> null", list.toString());
    }
    
    @Test
// Spring Framework Example
@RestController
@RequestMapping("/api")
public class UserController {
    @Autowired
    private UserRepository userRepository;
    @GetMapping("/users")
    public List<User> getAllUsers() {
        return userRepository.findAll();
    }
    @PostMapping("/users")
    public User createUser(@RequestBody User user) {
        return userRepository.save(user);
    }
}
// Spring Cloud Config Example
@Configuration
@EnableConfigServer
public class ConfigServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(ConfigServerApplication.class, args);
    }
}
// Spring Cloud Gateway Example
@Bean
public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
    return builder.routes()
        .route("path_route", r -> r.path("/get")
            .uri("http://httpbin.org"))
        .build();
}
// Spring Cloud Stream Example
@EnableBinding(Source.class)
public class MessageProducer {
    @Autowired
    private Source source;
    public void sendMessage(String message) {
        source.output().send(MessageBuilder.withPayload(message).build());
    }
}
// Spring Batch Example
@Configuration
@EnableBatchProcessing
public class BatchConfig {
    @Bean
    public Job importUserJob(JobBuilderFactory jobs, Step step1) {
        return jobs.get("importUserJob")
            .incrementer(new RunIdIncrementer())
            .flow(step1)
            .end()
            .build();
    }
}
// Spring Security Example
@Configuration
@EnableWebSecurity
public class WebSecurityConfig extends WebSecurityConfigurerAdapter {
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http
            .authorizeRequests()
                .antMatchers("/", "/home").permitAll()
                .anyRequest().authenticated()
                .and()
            .formLogin()
                .loginPage("/login")
                .permitAll()
                .and()
            .logout()
                .permitAll();
    }
}
// Spring Data REST Example
@RepositoryRestResource(collectionResourceRel = "people", path = "people")
public interface PersonRepository extends PagingAndSortingRepository<Person, Long> {
    List<Person> findByLastName(@Param("name") String name);
}
// Spring Integration Example
@MessagingGateway
public interface MessageGateway {
    @Gateway(requestChannel = "inputChannel")
    void sendMessage(String message);
}
// Spring Actuator Example
@Configuration
@Endpoint(id = "custom")
public class CustomEndpoint {
    @ReadOperation
    public String custom() {
        return "Custom Endpoint";
    }
}
// Spring Test Example
@RunWith(SpringRunner.class)
@SpringBootTest
public class ApplicationTests {
    @Test
    public void contextLoads() {
    }
}
// Microservice Example
@SpringBootApplication
@EnableDiscoveryClient
public class ProductServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(ProductServiceApplication.class, args);
    }
    @Bean
    @LoadBalanced
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
// Concurrent Programming
class ThreadSafeCounter {
    private int count = 0;
    private final Object lock = new Object();
    public void increment() {
        synchronized(lock) {
            count++;
        }
    }
    public int getCount() {
        synchronized(lock) {
            return count;
        }
    }
}
@Test
    void testPrepend() {
        list.prepend("A");
        assertEquals(1, list.size());
        assertEquals("A -> null", list.toString());
    }
}
// Main Class
public class Main {
    public static void main(String[] args) {
         Example
        LinkedList<Integer> list = new LinkedList<>();
        list.append(1);
        list.append(2);
        list.prepend(0);
        System.out.println("Linked List: " + list);
        
         Example
        int[] arr = {5, 3, 8, 4, 2};
        Sorting.bubbleSort(arr);
        System.out.print("Sorted Array: ");
        for (int num : arr) {
            System.out.print(num + " ");
        }
        System.out.println();
        System.out.println("Binary Search for 4: " + Sorting.binarySearch(arr, 4));
        
        // Multithreading Example
        Thread t1 = new Thread(new Worker("Worker 1"));
        Thread t2 = new Thread(new Worker("Worker 2"));
        t1.start();
        t2.start();
        
        // Spring Boot Microservices Example
        SpringApplication.run(DemoApplication.class, args);
        
        // Concurrent Programming Patterns
        ExecutorService executor = Executors.newFixedThreadPool(4);
        executor.submit(() -> System.out.println("Task 1 executed"));
        executor.submit(() -> System.out.println("Task 2 executed"));
        executor.shutdown();
        
        // Performance Optimization Example
        long start = System.nanoTime();
        // Optimized code here
        long duration = System.nanoTime() - start;
        System.out.println("Optimized operation took " + duration + " ns");
    }
}
// Additional Spring Boot Components
@Configuration
@EnableCaching
class CacheConfig {
    @Bean
    public CacheManager cacheManager() {
        return new ConcurrentMapCacheManager("products");
    }
}
// Advanced Concurrency Patterns
class ReadWriteLockExample {
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    private int value;
    
    public void write(int newValue) {
        lock.writeLock().lock();
        try {
            value = newValue;
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    public int read() {
        lock.readLock().lock();
        try {
            return value;
        } finally {
            lock.readLock().unlock();
        }
    }
}
// Reactive Programming Example
@RestController
class ReactiveController {
    @GetMapping("/reactive")
    public Mono<String> reactiveEndpoint() {
        return Mono.just("Reactive Response");
    }
}
// Distributed Tracing Example
@Configuration
class TracingConfig {
    @Bean
    public Tracer tracer() {
        return new OpenTelemetryTracer();
    }
}
// Circuit Breaker Pattern
@RestController
class ResilientController {
    @GetMapping("/resilient")
    @CircuitBreaker(fallbackMethod = "fallback")
    public String resilientEndpoint() {
        // Potentially failing operation
        return "Success";
    }
    
    public String fallback() {
        return "Fallback response";
    }
}