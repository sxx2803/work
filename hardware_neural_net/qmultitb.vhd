
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD;

entity testbench is
end entity;

architecture testbench of testbench is
  
component qMultiplier
  generic(n            : Integer :=7;
          m            : Integer :=1);
  port   (A,B          : in std_logic_vector((n+m+1)downto 1);
          output       : out std_logic_vector((n+m+1)downto 1));
end component qMultiplier;

signal sA,sB: std_logic_vector(9 downto 1);
signal soutput: std_logic_vector(9 downto 1);

begin
  uut1: qMultiplier port map(A=>sA, B=>sB, output=>soutput);
    process
      begin
      wait for 10 ns;
      sA <= "101000001"; -- -1.4921875
      sB <= "111101111"; -- -0.1328125 - 000011000
      wait for 10 ns;
      sA <= "101001001"; -- -1.4296875
      sB <= "110111011"; -- -0.5390625 - 001100010
      wait for 10 ns;
      sA <= "011010101"; -- 1.6640625
      sB <= "010000001"; -- 1.0078125  - 011010110
      wait for 10 ns;
      sA <= "011111111"; -- 1.9921875
      sB <= "111110100"; -- -0.09375   - 111101001
      wait for 10 ns;
      sA <= "100000000"; -- -2.0
      sB <= "001001010"; -- 578125     - 101101100
      wait for 10 ns;
      sA <= "100000000"; -- -2.0
      sB <= "011111111"; -- 1.9921875  - 100000010
	  wait for 10 ns;
      sA <= "010000000"; -- 1.0
      sB <= "111111111"; -- 1.9921875  - 100000010
      wait;
    end process;
end architecture testbench;
