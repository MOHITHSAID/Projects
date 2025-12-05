class RecentCounter {
    
    Queue<Integer>q1;
    public RecentCounter() {
        
        q1=new LinkedList<>();
    }
    
    public int ping(int t) {
        
        q1.offer(t);
        while(q1.peek()<(t-3000)){
            q1.poll();
        }
        return q1.size();
    }
}
