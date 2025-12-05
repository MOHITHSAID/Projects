class Solution {
    public int[] deckRevealedIncreasing(int[] deck) {
        int n=deck.length;
        Deque<Integer>dq=new ArrayDeque<>();
        Arrays.sort(deck);
        for(int i=n-1;i>=0;i--){
            if(!dq.isEmpty()){
                int val=dq.removeLast();
                dq.addFirst(val);
            }
            dq.addFirst(deck[i]);
        }
        int result[]=new int[n];
        int i=0;
        for(int val:dq){
            result[i]=val;
            i++;
        }

    return result;
    }
}
