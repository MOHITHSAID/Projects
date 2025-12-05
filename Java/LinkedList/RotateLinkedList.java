//Class and implementation of LinkedList in the file LinkedList
public Node rotateRight(int k) { 
        if(head==null) return null;
        if(head.next==null) return head;
        Node tail=head;
        int length=0;
        Node n2=head;
        Node temp=head;
        while(n2!=null){
            length++;
            n2=n2.next;
        }
        while(tail.next!=null){
           
            tail=tail.next;
            
        }
        tail.next=head;
        k=k%length;
        int diff=length-k;
        
        for(int i=0;i<diff-1;i++){
            temp=temp.next;
        }
        Node head2=temp.next;
        temp.next=null;
        return head2;
    }
