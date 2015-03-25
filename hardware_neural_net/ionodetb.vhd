
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD;

entity ionodetb is
end entity ionodetb;

architecture ionodetb of ionodetb is
  
component ionode
	generic (
		numInputs : integer := 3;
		n         : integer := 7;
		m         : integer := 1
	);
	port (
		clk      :	in std_logic;
		en       :	in std_logic;
		weights  :	in std_logic_vector(numInputs*(n+m+1) downto 1);
		xIn      :	in std_logic_vector(numInputs*(n+m+1) downto 1);
		yOut     :	out std_logic_vector(n+m+1 downto 1)
	);
end component ionode;

signal	sclk      :	std_logic := '0';
signal	sen       :	std_logic;
signal	sweights  :	std_logic_vector(2*(9) downto 1);
signal	sxIn      :	std_logic_vector(2*(9) downto 1);
signal	syOut     :	std_logic_vector(9 downto 1);

begin
	uut1: ionode	generic map(numInputs => 2, m => 1, n=>7)
					port map(clk=>sclk, en=>sen, weights=>sweights, xIn=>sxIn, yOut=>syOut);
	
	sclk <= not sclk after 10 ns;
    sen <= '1';
    sweights <= "010000000010000000";
    sxIn <= "101000001010101010";
end architecture ionodetb;