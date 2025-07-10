// SPDX-License-Identifier: MIT

pragma solidity ^0.8.13;

library TransferHelper {
    function safeTransferFrom(
        address token,
        address from,
        address to,
        uint256 value
    ) internal {
        (bool success, bytes memory data) = token.call(
            abi.encodeWithSelector(0x23b872dd, from, to, value)
        );

        require(
            success && (data.length == 0 || abi.decode(data, (bool))),
            "TransferHelper: TRANSFER_FROM_FAILED "
        );
    }
}

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);

    function allowance(address _owner, address spender)
        external
        view
        returns (uint256);

    function transfer(address recipient, uint256 amount)
        external
        returns (bool);
}

abstract contract Ownable {
    address private _owner;

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    /**
     * @dev Initializes the contract setting the deployer with given initialOwner.
     */
    constructor(address initialOwner) {
        _setOwner(initialOwner);
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view virtual returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        require(owner() == msg.sender, "Ownable: caller is not the owner");
        _;
    }

    /**
     * @dev Leaves the contract without owner. It will not be possible to call
     * `onlyOwner` functions anymore. Can only be called by the current owner.
     *
     * NOTE: Renouncing ownership will leave the contract without an owner,
     * thereby removing any functionality that is only available to the owner.
     */
    function renounceOwnership() public virtual onlyOwner {
        _setOwner(address(0));
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(
            newOwner != address(0),
            "Ownable: new owner is the zero address"
        );
        _setOwner(newOwner);
    }

    function _setOwner(address newOwner) private {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

contract WPEPEP is Ownable {
    string public name = "Wrapped PEPEP";
    string public symbol = "WPEPEP";
    uint8 public decimals = 18;

    address public pepep = 0x90f9FC2792358aa70C5a10b26e10830883018022;

    uint256 public deadline = 1761670800; // 8 months from now

    event Approval(address indexed src, address indexed guy, uint256 wad);
    event Transfer(address indexed src, address indexed dst, uint256 wad);
    event Deposit(address indexed dst, uint256 wad);
    event Withdrawal(address indexed src, uint256 wad);

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    constructor() Ownable(msg.sender) {}

    function deposit(uint256 wad, address dst) public {
        require(
            IERC20(pepep).allowance(msg.sender, address(this)) >= wad,
            "WPEPEP: not enough allowance"
        );
        TransferHelper.safeTransferFrom(pepep, msg.sender, address(this), wad);
        if (dst == address(0)) {
            dst = msg.sender;
        }
        balanceOf[dst] += wad;
        emit Deposit(dst, wad);
    }

    function withdraw(uint256 wad) public {
        require(block.timestamp > deadline, "WPEPEP: not due yet");
        require(balanceOf[msg.sender] >= wad);
        balanceOf[msg.sender] -= wad;
        bool result = IERC20(pepep).transfer(msg.sender, wad);
        require(result, "WPEPEP: transfer failed");
        emit Withdrawal(msg.sender, wad);
    }

    function totalSupply() public view returns (uint256) {
        return IERC20(pepep).balanceOf(address(this));
    }

    function approve(address guy, uint256 wad) public returns (bool) {
        allowance[msg.sender][guy] = wad;
        emit Approval(msg.sender, guy, wad);
        return true;
    }

    function transfer(address dst, uint256 wad) public returns (bool) {
        return transferFrom(msg.sender, dst, wad);
    }

    function transferFrom(
        address src,
        address dst,
        uint256 wad
    ) public returns (bool) {
        require(balanceOf[src] >= wad);

        if (src != msg.sender) {
            require(allowance[src][msg.sender] >= wad);
            allowance[src][msg.sender] -= wad;
        }

        balanceOf[src] -= wad;
        balanceOf[dst] += wad;
        emit Transfer(src, dst, wad);
        return true;
    }

    function setDeadline(uint256 _dl) public onlyOwner {
        deadline = _dl;
    }

    function emergency() public onlyOwner {
        IERC20(pepep).transfer(
            msg.sender,
            IERC20(pepep).balanceOf(address(this))
        );
    }
}