pragma solidity ^0.8.0;

contract HelloWorld {
    string public message;
    event NewMessage(string message);

    Bridge public bridge;
    constructor(address bridgeAddress) {
        message = "Hello, World!";
        bridge = Bridge(bridgeAddress);
    }

    // Function to set a new message
    function setMessage(string calldata newMessage) external {
        message = newMessage;
        emit NewMessage(newMessage);
    }

    // Function to get the current message
    function getMessage() external view returns (string memory) {
        return message;
    }




    function portMessage(
        address sender,
        CrossChainData memory data,
        string memory _startChain
    ) external {
        require(data.addresses.length > 0, "Address array empty");
        require(data.strings.length > 0, "String array empty");

        // Set the message to the first string in the array
        message = data.strings[0];

        emit NewMessage(data.strings[0]);
    }


    function sendMessage(address to, string  calldata chain) public payable returns (bool) {
        require(msg.value > 0, "Fee amount must be greater than 0");
        address[] memory addresses = new address[](1);
        uint256[] memory integers = new uint256[](1);
        string[] memory strings = new string[](1);
        bool[] memory bools = new bool[](1);

        strings[0] = message;

        require(addresses.length <= 5, "Addresses array length must be <= 5");
        require(integers.length <= 5, "Integers array length must be <= 5");
        require(strings.length <= 5, "Strings array length must be <= 5");
        require(bools.length <= 5, "Bools array length must be <= 5");

        // Prepare CrossChainData
        CrossChainData memory data = CrossChainData({
            addresses: addresses,
            integers: integers,
            strings: strings,
            bools: bools
        });

        // Call the bridge's outboundMessage
        bridge.outboundMessage{value: msg.value}(
            address(this),
            to,
            data,
            chain
        );

        return true;
    }
}

struct CrossChainData {
        address[] addresses;
        uint256[] integers;
        string[] strings;
        bool[] bools;
    }
    
// Define the Bridge interface
interface Bridge {
    function outboundMessage(
        address from,
        address to,
        CrossChainData calldata data,
        string calldata chain
    ) external payable;
}