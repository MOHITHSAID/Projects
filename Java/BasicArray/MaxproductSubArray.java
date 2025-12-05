class Solution {
    public int maxProduct(int[] nums) {
        int size=nums.length;
        int maxValue=nums[0];
        int minValue=nums[0];
        int ans=nums[0];
        for(int i=1;i<size;i++){
            int temp=maxValue;
            int val=nums[i];
            maxValue=Math.max(val,Math.max(maxValue*val,minValue*val));
            minValue=Math.min(val,Math.min(temp*val,minValue*val));
            ans=Math.max(ans,maxValue);
        }
        return ans;
    }
}
