class Solution {
    public int maximumProduct(int[] nums, int k) {
        int MOD = 1_000_000_007;
        PriorityQueue<Integer>pq=new PriorityQueue<>();
        int n=nums.length;
        for(int i=0;i<n;i++){
            pq.offer(nums[i]);
        }
        while(!pq.isEmpty() && (k>0)){
            int val=pq.poll();
            pq.offer(val+1);           
            k--;
        }

        long product=1;
        while(!pq.isEmpty()){
            product = (product * pq.poll()) % MOD;
        }
        
        return (int)product;
    }
}
