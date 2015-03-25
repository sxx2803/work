library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD;

entity neurontb is
end entity;

architecture neurontb of neurontb is
  
component neuron
	generic(numInputs: integer := 10;
			n: integer := 1;
			m: integer := 7);
	port(clk: in std_logic;
		en:		in std_logic;
		dataIn: in std_logic_vector(numInputs*(n+m+1) downto 1);
		dataOut: out std_logic_vector(n+m+1 downto 1));
end component;

signal sclk: std_logic := '0';
signal sdataIn: std_logic_vector(10*(9) downto 1);
signal sdataOut: std_logic_vector(9 downto 1);

begin
	uut1: neuron port map(clk=>sclk, dataIn=>sdataIn, dataOut=>sdataOut, en=>'1');
  
	sclk <= not sclk after 10 ns;
	sdataIn <= "110000000000110101010000000110000000000000001000000000110000000110000000110000000010000000";
  
end architecture neurontb;