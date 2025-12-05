class MinStack {
    Stack<Integer> s1;
    Stack<Integer> minstack;

    public MinStack() {
        s1 = new Stack<>();
        minstack = new Stack<>();
    }
    
    public void push(int val) {
        s1.push(val);
        if (minstack.isEmpty() || val <= minstack.peek()) {
            minstack.push(val);
        }
    }
    
    public void pop() {
        int removed = s1.pop();
        if (removed == minstack.peek()) {
            minstack.pop();
        }
    }
    
    public int top() {
        return s1.peek();
    }
    
    public int getMin() {
        return minstack.peek();
    }
}
