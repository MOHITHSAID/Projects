class Solution {
    public int[] maxSubsequence(int[] nums, int k) {
       PriorityQueue<Integer>pq=new PriorityQueue<>((a,b)->(b-a));
        if(nums.length==k) return nums;
        for(int i=0;i<nums.length;i++){
            pq.add(nums[i]);
        }
        int arr[]=new int[k];
        for(int i=0;i<k;i++){
            arr[i]=pq.poll();
        }
        int index[]=new int[k];
        int idx=0;

        for(int i=0;(i<nums.length)&&(idx<k);i++){
           for(int j=0;j<k;j++){
            if(nums[i]==arr[j]){
                index[idx]=i;
                idx++;
                arr[j] = Integer.MIN_VALUE; // mark used
                break;
            }
           }
        }

        Arrays.sort(index);
        int result[]=new int[k];
        for(int i=0;i<k;i++){
            result[i]=nums[index[i]];
        }

        return result;
    }
}
