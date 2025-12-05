Circular Queue:
class MyCircularQueue {
    int queue[];
    int capacity;
    int rear=-1;
    int front=-1;
    
    public MyCircularQueue(int k) {
        capacity=k;
        queue=new int[capacity];
    }
    
    public boolean enQueue(int value) {
        if(front==(rear+1)%capacity){
            return false;
        }
        else if(front==-1){
            front=0;
            rear=0;
            queue[rear]=value;
            return true;
        }
        else{
            rear=(rear+1)%capacity;
            queue[rear]=value;
            return true;
        }
    }
    
    public boolean deQueue() {
        if(front==-1) return false;
        if(front==rear){
            front=-1;
            rear=-1;
            return true;
        }
        else{
            front=(front+1)%capacity;
            return true;
        }

    }
    
    public int Front() {
        if(front==-1) return -1;
        else{
            return queue[front];
        }
    }
    
    public int Rear() {
        if(isEmpty()) return -1;
        else{
            return queue[rear];
        }
    }
    
    public boolean isEmpty() {
        return (front==-1);
    }
    
    public boolean isFull() {
        return(front==(rear+1)%capacity);
    }
}
