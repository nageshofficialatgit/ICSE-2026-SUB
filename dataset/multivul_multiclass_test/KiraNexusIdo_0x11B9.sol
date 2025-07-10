// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (utils/introspection/IERC165.sol)

pragma solidity ^0.8.13;

// OpenZeppelin Contracts v4.4.1 (utils/Context.sol)

interface IBEP20 {
    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address _owner, address spender)
        external
        view
        returns (uint256);
}

/**
 * @dev Provides information about the current execution context, including the
 * sender of the transaction and its data. While these are generally available
 * via msg.sender and msg.data, they should not be accessed in such a direct
 * manner, since when dealing with meta-transactions the account sending and
 * paying for execution may not be the actual sender (as far as an application
 * is concerned).
 *
 * This contract is only required for intermediate, library-like contracts.
 */
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

// OpenZeppelin Contracts v4.4.1 (access/Ownable.sol)

/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * By default, the owner account will be the one that deploys the contract. This
 * can later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
abstract contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    /**
     * @dev Initializes the contract setting the deployer as the initial owner.
     */
    constructor() {
        _transferOwnership(_msgSender());
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
        require(owner() == _msgSender(), "You are not the owner");
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
        _transferOwnership(address(0));
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
        _transferOwnership(newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

contract VerifySignature {
    function getMessageHash(
        address _to,
        uint256 _payment,
        uint256 _amount,
        uint256 _usdAmount,
        uint256 _amountOut,
        uint256 _nonce
    ) public pure returns (bytes32) {
        return
            keccak256(
                abi.encodePacked(
                    _to,
                    _payment,
                    _amount,
                    _usdAmount,
                    _amountOut,
                    _nonce
                )
            );
    }

    function getEthSignedMessageHash(bytes32 _messageHash)
        public
        pure
        returns (bytes32)
    {
        /*
        Signature is produced by signing a keccak256 hash with the following format:
        "\x19Ethereum Signed Message\n" + len(msg) + msg
        */
        return
            keccak256(
                abi.encodePacked(
                    "\x19Ethereum Signed Message:\n32",
                    _messageHash
                )
            );
    }

    function verify(
        address _signer,
        address _to,
        uint256 _payment,
        uint256 _amount,
        uint256 _usdAmount,
        uint256 _amountOut,
        uint256 _nonce,
        bytes memory signature
    ) public pure returns (bool) {
        bytes32 messageHash = getMessageHash(
            _to,
            _payment,
            _amount,
            _usdAmount,
            _amountOut,
            _nonce
        );
        bytes32 ethSignedMessageHash = getEthSignedMessageHash(messageHash);

        return recoverSigner(ethSignedMessageHash, signature) == _signer;
    }

    function recoverSigner(
        bytes32 _ethSignedMessageHash,
        bytes memory _signature
    ) public pure returns (address) {
        (bytes32 r, bytes32 s, uint8 v) = splitSignature(_signature);

        return ecrecover(_ethSignedMessageHash, v, r, s);
    }

    function splitSignature(bytes memory sig)
        public
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

library TransferHelper {
    function safeTransferETH(address to, uint256 value) internal {
        (bool success, ) = to.call{value: value}(new bytes(0));
        require(success, "TransferHelper: ETH_TRANSFER_FAILED");
    }

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

contract KiraNexusIdo is Ownable, VerifySignature {
    receive() external payable {}

    event Buy(
        address indexed sender,
        address indexed rerecipient,
        uint256 amount,
        uint256 tokenAmount,
        uint256 refTokenAmount,
        address indexed ref
    );

    address public devAddress = 0x1888e50A1dF38597855aF51B13BC2e83CE2D2896;
    // uint256 public refRate = 5;
    uint256 public refTokenRate = 5;

    address public usdt = 0xdAC17F958D2ee523a2206206994597C13D831ec7;
    address public usdc = address(0);

    address public tokenAddress = 0xCDb02507CB3C01c9B260e2a34bFB5c0cCdc0Ad45;

    mapping(uint256 => bool) public usedNonce;

    struct Owner {
        uint256 tokenAmount;
        uint256 usdAmount;
    }

    mapping(address => Owner) public owners;

    uint256 public totalToken;
    uint256 public totalUsd;

    constructor() {}

    function withdraw() public onlyOwner {
        require(address(this).balance > 0, "Invalid amount");
        TransferHelper.safeTransferETH(msg.sender, address(this).balance);
    }

    function setDevAddress(address _devAddress) public onlyOwner {
        devAddress = _devAddress;
    }

    function setUsdt(address _usdt) public onlyOwner {
        usdt = _usdt;
    }

    function setUsdc(address _usdc) public onlyOwner {
        usdc = _usdc;
    }

    // function setRefRate(uint256 _refRate) public onlyOwner {
    //     refRate = _refRate;
    // }

    function setRefTokenRate(uint256 _refTokenRate) public onlyOwner {
        refTokenRate = _refTokenRate;
    }

    function safeTransfer(
        uint256 _amount,
        uint256 _payment,
        address _sender,
        address _receiver
    ) internal {
        if (_payment == 1) {
            TransferHelper.safeTransferETH(_receiver, _amount);
        } else {
            TransferHelper.safeTransferFrom(
                _payment == 2 ? usdt : usdc,
                _sender,
                _receiver,
                _amount
            );
        }
    }

    function transfer(
        address _ref,
        uint256 _amount,
        uint256 _payment,
        uint256 _amountOut,
        address _recipient,
        uint256 refTokenAmount
    ) internal {
        safeTransfer(_amount, _payment, msg.sender, devAddress);
        // transfer token to _ref
        if (_ref != address(0) && refTokenAmount > 0 && _ref != msg.sender) {
            TransferHelper.safeTransferFrom(
                tokenAddress,
                owner(),
                _ref,
                refTokenAmount
            );
        }

        TransferHelper.safeTransferFrom(
            tokenAddress,
            owner(),
            _recipient,
            _amountOut
        );
    }

    function buyKLAI(
        address _ref,
        uint256 _amount,
        uint256 _usdAmount,
        uint256 _payment,
        uint256 _amountOut,
        uint256 _nonce,
        address _recipient,
        bytes calldata _signature
    ) public payable {
        require(
            verify(
                owner(),
                msg.sender,
                _payment,
                _amount,
                _usdAmount,
                _amountOut,
                _nonce,
                _signature
            ) && usedNonce[_nonce] == false,
            "Invalid signature"
        );

        _recipient = _recipient == address(0) ? msg.sender : _recipient;

        owners[_recipient].usdAmount += _usdAmount;
        owners[_recipient].tokenAmount += _amountOut;

        totalToken += _amountOut;
        totalUsd += _usdAmount;

        uint256 refTokenAmount = 0;
        if (_ref != address(0) && _ref != msg.sender) {
            refTokenAmount = (_amountOut * refTokenRate) / 10000;
        }

        transfer(_ref, _amount, _payment, _amountOut, _recipient, refTokenAmount);
        usedNonce[_nonce] == true;

        emit Buy(
            msg.sender,
            _recipient,
            _usdAmount,
            _amountOut,
            refTokenAmount,
            _ref
        );
    }
}