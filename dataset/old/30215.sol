pragma solidity ^0.4.18;
contract HelloWorld{
    string input = "Hello world.";
    function sayHello() view public returns (string) {
        return input;
    }
    function setNewGreeting(string greeting) public {
        input = greeting;
    }
}