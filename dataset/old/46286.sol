pragma solidity ^0.4.21;
interface ERC20 {
  function transfer( address to, uint256 value ) external;
}
contract Airdropper {
  function airdrop( address tokAddr,
                    address[] dests,
                    uint[] quantities ) public returns (uint) {
    for (uint ii = 0; ii < dests.length; ii++) {
      ERC20(tokAddr).transfer( dests[ii], quantities[ii] );
    }
    return ii;
  }
}