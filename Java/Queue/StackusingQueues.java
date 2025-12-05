class MyStack {
    Queue<Integer> q = new LinkedList<>();
    public void push(int x) {
        q.offer(x);
        int Qsize=q.size();
        for(int i=0;i<Qsize-1;i++){
            q.offer(q.peek());
            q.poll();
        }
    }
    public int pop() { return q.poll(); }
    public int top() { return q.peek(); }
    public boolean empty() { return q.isEmpty(); }
}
