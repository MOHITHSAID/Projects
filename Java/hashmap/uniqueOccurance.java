//1207 unique number of Occurance
class Solution {
    public boolean uniqueOccurrences(int[] arr) {
        Map<Integer,Integer>hm=new HashMap<>();
        for(int i:arr){
            hm.put(i,hm.getOrDefault(i,0)+1);
        }
        Set<Integer>hs=new HashSet<>(hm.values());
        return hs.size()==hm.size();


    }
}
