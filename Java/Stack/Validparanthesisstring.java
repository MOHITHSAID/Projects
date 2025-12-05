class Solution {
    public boolean checkValidString(String s) {
        int mincount=0;
        int maxcount=0;
        for(char c:s.toCharArray()){
            if(c=='('){
                mincount++;
                maxcount++;
            }
            if(c==')'){
                mincount--;
                maxcount--;
            }
            if(c=='*'){
                mincount--;
                maxcount++;
            }
            if(maxcount<0) return false;
            if(mincount<0) {
                mincount=0;
            }
        }
        return (mincount==0);
    }
}
