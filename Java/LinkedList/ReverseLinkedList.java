    public Node reverseList() {
        if(head==null) return null;
        Node curr=head;
        Node n1=head;
        int len=0;
        while(head!=null){
            head=head.next;
            len=len+1;
        }
        int arr[]=new int[len];
        int i=0;
        while((curr!=null)&&(i<len)){
            arr[i]=curr.data;
            curr=curr.next;
            i=i+1;
        }
        Node n2=n1;
        for(i=0;i<len;i++){
            n1.data=arr[len-i-1];
            n1=n1.next;
        }
        return n2;
    }
