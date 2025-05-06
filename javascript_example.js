// Data Structures
class LinkedListNode {
    constructor(value, next = null) {
        this.value = value;
        this.next = next;
    }
}

// Frontend Framework Examples
class ReactComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = { count: 0 };
    }
    
    increment = () => {
        this.setState(prevState => ({ count: prevState.count + 1 }));
    }
    
    render() {
        return (
            <div>
                <p>Count: {this.state.count}</p>
                <button onClick={this.increment}>Increment</button>
            </div>
        );
    }
}

// Node.js Backend Example
const express = require('express');
const app = express();

app.use(express.json());

app.get('/api/users', (req, res) => {
    res.json([{ id: 1, name: 'John' }, { id: 2, name: 'Jane' }]);
});

app.post('/api/users', (req, res) => {
    const newUser = req.body;
    // Save to database
    res.status(201).json(newUser);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

class LinkedList {
    constructor() {
        this.head = null;
        this.tail = null;
        this.length = 0;
    }

    prepend(value) {
        const newNode = new LinkedListNode(value, this.head);
        this.head = newNode;
        if (!this.tail) this.tail = newNode;
        this.length++;
        return this;
    }

    append(value) {
        const newNode = new LinkedListNode(value);
        if (!this.head) {
            this.head = newNode;
            this.tail = newNode;
            return this;
        }
        this.tail.next = newNode;
        this.tail = newNode;
        this.length++;
        return this;
    }

    delete(value) {
        if (!this.head) return null;
        let deletedNode = null;
        while (this.head && this.head.value === value) {
            deletedNode = this.head;
            this.head = this.head.next;
            this.length--;
        }
        let currentNode = this.head;
        if (currentNode !== null) {
            while (currentNode.next) {
                if (currentNode.next.value === value) {
                    deletedNode = currentNode.next;
                    currentNode.next = currentNode.next.next;
                    this.length--;
                } else {
                    currentNode = currentNode.next;
                }
            }
        }
        if (this.tail && this.tail.value === value) {
            this.tail = currentNode;
        }
        return deletedNode;
    }

    find(value) {
        if (!this.head) return null;
        let currentNode = this.head;
        while (currentNode) {
            if (currentNode.value === value) {
                return currentNode;
            }
            currentNode = currentNode.next;
        }
        return null;
    }

    toArray() {
        const nodes = [];
        let currentNode = this.head;
        while (currentNode) {
            nodes.push(currentNode.value);
            currentNode = currentNode.next;
        }
        return nodes;
    }
}

// Algorithms
function bubbleSort(arr) {
    let len = arr.length;
    for (let i = 0; i < len; i++) {
        for (let j = 0; j < len - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
            }
        }
    }
    return arr;
}

function binarySearch(sortedArray, key) {
    let start = 0;
    let end = sortedArray.length - 1;
    while (start <= end) {
        let middle = Math.floor((start + end) / 2);
        if (sortedArray[middle] === key) return middle;
        else if (sortedArray[middle] < key) start = middle + 1;
        else end = middle - 1;
    }
    return -1;
}

// Utility Functions
function debounce(func, wait, immediate) {
    let timeout;
    return function() {
        const context = this, args = arguments;
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(context, args);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Async Operations
async function fetchData() {
    try {
        const response = await fetch('https://api.example.com/data');
        const data = await response.json();
        
        const processed = data.map(item => ({
            ...item,
            createdAt: new Date(item.createdAt),
            value: parseFloat(item.value)
        }));
        
        return processed;
    } catch (error) {
        console.error('Error fetching data:', error);
        throw error;
    }
}

async function fetchWithRetry(url, options = {}, retries = 3) {
    try {
        const response = await fetch(url, options);
        if (!response.ok) throw new Error(response.statusText);
        return await response.json();
    } catch (err) {
        if (retries <= 0) throw err;
        await new Promise(resolve => setTimeout(resolve, 1000));
        return await fetchWithRetry(url, options, retries - 1);
    }
}

// DOM Utilities
function createElement(tag, attributes = {}, children = []) {
    const element = document.createElement(tag);
    Object.keys(attributes).forEach(attr => {
        element.setAttribute(attr, attributes[attr]);
    });
    children.forEach(child => {
        if (typeof child === 'string') {
            element.appendChild(document.createTextNode(child));
        } else {
            element.appendChild(child);
        }
    });
    return element;
}

function delegateEvent(parent, eventName, selector, handler) {
    parent.addEventListener(eventName, function(event) {
        let target = event.target;
        while (target && target !== parent) {
            if (target.matches(selector)) {
                handler.call(target, event);
                break;
            }
            target = target.parentNode;
        }
    });
}

// Functional Programming
function compose(...fns) {
    return function(initialValue) {
        return fns.reduceRight((value, fn) => fn(value), initialValue);
    };
}

function curry(fn) {
    return function curried(...args) {
        if (args.length >= fn.length) {
            return fn.apply(this, args);
        } else {
            return function(...args2) {
                return curried.apply(this, args.concat(args2));
            };
        }
    };
}

// Event Handling
class EventEmitter {
    constructor() {
        this.events = {};
    }

    on(eventName, listener) {
        if (!this.events[eventName]) {
            this.events[eventName] = [];
        }
        this.events[eventName].push(listener);
        return this;
    }

    emit(eventName, ...args) {
        if (!this.events[eventName]) return false;
        this.events[eventName].forEach(listener => {
            listener.apply(this, args);
        });
        return true;
    }

    off(eventName, listener) {
        if (!this.events[eventName]) return this;
        if (!listener) {
            delete this.events[eventName];
            return this;
        }
        this.events[eventName] = this.events[eventName].filter(
            l => l !== listener
        );
        return this;
    }
}

// Testing Utilities
function describe(description, fn) {
    console.log(`\n${description}`);
    fn();
}

function it(description, fn) {
    try {
        fn();
        console.log(`  ✓ ${description}`);
    } catch (error) {
        console.error(`  ✗ ${description}`);
        console.error(error);
    }
}

function expect(actual) {
    return {
        toBe(expected) {
            if (actual !== expected) {
                throw new Error(`Expected ${expected} but got ${actual}`);
            }
        },
        toEqual(expected) {
            if (JSON.stringify(actual) !== JSON.stringify(expected)) {
                throw new Error(`Expected ${JSON.stringify(expected)} but got ${JSON.stringify(actual)}`);
            }
        },
        toThrow() {
            try {
                actual();
                throw new Error('Expected function to throw but it did not');
            } catch (error) {
                return;
            }
        }
    };
}

// React Components
class Counter extends React.Component {
    constructor(props) {
        super(props);
        this.state = { count: 0 };
    }

    increment = () => {
        this.setState(prevState => ({ count: prevState.count + 1 }));
    };

    decrement = () => {
        this.setState(prevState => ({ count: prevState.count - 1 }));
    };

    render() {
        return (
            <div>
                <h2>Counter: {this.state.count}</h2>
                <button onClick={this.increment}>+</button>
                <button onClick={this.decrement}>-</button>
            </div>
        );
    }
}

// State Management with Redux
const counterReducer = (state = { count: 0 }, action) => {
    switch (action.type) {
        case 'INCREMENT':
            return { count: state.count + 1 };
        case 'DECREMENT':
            return { count: state.count - 1 };
        default:
            return state;
    }
};

const store = Redux.createStore(counterReducer);

// Web API Wrapper
class ApiService {
    static async get(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    }

    static async post(url, data) {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    }
}