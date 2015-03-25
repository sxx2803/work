
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD;

entity shiftRegtb is
end entity;

architecture shiftRegtb of shiftRegtb is
  
component weightShift
	
	generic(nInputs   : integer := 8);
	port (
		clk     :	in std_logic;
		SE      :	in std_logic;
		WSI     :	in std_logic;
		output  :	out std_logic_vector(nInputs downto 1)
	);
end component weightShift;

signal		sclk     :	 std_logic := '0';
signal		sSE      :	 std_logic := '0';
signal		sWSI     :	 std_logic;
signal		soutput  :	 std_logic_vector(8 downto 1);

begin
	uut1: weightShift port map(clk=>sclk, SE=>sSE, WSI=>sWSI, output=>soutput);
	
	sclk <= not sclk after 10 ns;
    process
      begin
      wait for 20 ns;
      sWSI <= '1';
      wait for 20 ns;
      sWSI <= '0';
      wait for 20 ns;
      sWSI <= '0';
      wait for 20 ns;
      sWSI <= '1';
      wait for 20 ns;
      sSE <= '1';
      sWSI <= '0';
      wait for 20 ns;
      sWSI <= '1';
      wait for 20 ns;
      sWSI <= '1';
      wait for 20 ns;
      sWSI <= '0';
      wait for 20 ns;
      sWSI <= '1';
      wait for 20 ns;
      sWSI <= '1';
      wait for 20 ns;
      sWSI <= '0';
      wait for 20 ns;
      sWSI <= '0';
      wait for 20 ns;
      sSE <= '0';
      wait;
    end process;
end architecture shiftRegtb;