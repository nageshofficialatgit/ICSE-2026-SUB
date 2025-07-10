pragma solidity ^0.4.15;
contract Burner {
    function tokenFallback(address , uint , bytes ) returns (bool result) {
        return true;
    }
}