public boolean isPalindrome() {
        Node tail=head;
        int len=0;
        while(tail!=null){
            len=len+1;
            tail=tail.next;
        }
        int i=0;
        int arr[]=new int[len];
        while(head!=null && i<len){
            arr[i]=head.data;
            i++;
            head=head.next;
        }

        for(int j=0;j<arr.length/2;j++){
            if(arr[j]!=arr[arr.length-1-j]) return false;

        }
        return true;
        
    }
