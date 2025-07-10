pragma solidity ^0.4.19;
contract mortal {
    address owner;
    function mortal() { owner = msg.sender; }
    function kill() { if (msg.sender == owner) selfdestruct(owner); }
}
contract ObjectiveStorage is mortal {
    address creator;
    string objective;
    function ObjectiveStorage(string _objective) public
    {
        creator = msg.sender;
        objective = _objective;
    }
    function getObjective() public constant returns (string) {
        return objective;
    }
}