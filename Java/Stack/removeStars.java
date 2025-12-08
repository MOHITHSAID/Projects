class Solution {        //LeetCode 2390
    public String removeStars(String s) {
        Stack<Character> s1=new Stack<>();

        for(char c:s.toCharArray()){
            if(c!='*'){
                s1.push(c);
            }
            else{
                if(!s1.isEmpty()){
                    s1.pop();
                }
            }
        }
        StringBuffer res=new StringBuffer();
        while(!s1.isEmpty()){
            res.append(s1.pop());
            //s1.pop();
        }
        return res.reverse().toString();
    }
}
